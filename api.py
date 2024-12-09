from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import argparse
from typing import Optional
import os
from datetime import datetime
import aiofiles
import threading
import shutil
import time
import multiprocessing
from functools import partial

from omni_animate.pipelines.pipeline_animate_master import AnimateMasterPipeline


# 数据模型
class AnimationRequest(BaseModel):
    token: str
    height: int = 1024
    width: int = 576
    keep_ratio: bool = True
    keep_ref_dim: bool = False
    stride: int = 1
    seed: int = 1234
    num_inference_steps: int = 20
    guidance_scale: float = 3.0


class TaskStatus(BaseModel):
    task_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None


# 任务处理器
class TaskProcessor:
    def __init__(self, num_workers: int = 1, token: str = ""):
        self.task_queue = multiprocessing.Queue()
        self.results = {}
        self.num_workers = num_workers
        self.workers = []
        self.running = True
        self.token = token

    def start(self):
        for _ in range(self.num_workers):
            worker = multiprocessing.Process(target=self._worker_process)
            worker.start()
            self.workers.append(worker)

        self.result_thread = threading.Thread(target=self._collect_results)
        self.result_thread.start()

    def stop(self):
        self.running = False
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()
        self.result_thread.join()

    def _worker_process(self):
        pipe = AnimateMasterPipeline(token=self.token)
        while True:
            task = self.task_queue.get()
            if task is None:
                break

            task_id, ref_image_path, drive_video_path, params = task
            try:
                output_path = pipe(
                    ref_image_path=ref_image_path,
                    src_video_path=drive_video_path,
                    **params
                )

                final_output_path = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
                shutil.move(output_path, final_output_path)

                self.results[task_id] = {
                    "status": "completed",
                    "output_path": final_output_path
                }

            except Exception as e:
                self.results[task_id] = {
                    "status": "failed",
                    "error": str(e)
                }
            finally:
                for path in [ref_image_path, drive_video_path]:
                    if os.path.exists(path):
                        os.remove(path)

    def _collect_results(self):
        while self.running:
            for task_id, result in self.results.items():
                if task_id in tasks:
                    tasks[task_id].update(result)
            time.sleep(0.1)

    def add_task(self, task_id: str, ref_image_path: str,
                 drive_video_path: str, params: dict):
        self.task_queue.put((task_id, ref_image_path, drive_video_path, params))


# 全局变量
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "outputs/api"
tasks = {}
task_processor = None

# FastAPI 应用
app = FastAPI(title="OmniAnimate API")

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_args():
    parser = argparse.ArgumentParser(description='OmniAnimate API')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='服务器IP地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=6006,
                        help='服务器端口 (默认: 6006)')
    parser.add_argument('--debug', action='store_true',
                        help='是否启用调试模式 (默认: False)')
    parser.add_argument('--token', type=str, required=True,
                        help='用于动画管道的令牌')
    return parser.parse_args()


def create_startup_event(token):
    @app.on_event("startup")
    async def startup_event():
        global task_processor
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        task_processor = TaskProcessor(num_workers=1, token=token)
        task_processor.start()


@app.on_event("shutdown")
async def shutdown_event():
    if task_processor:
        task_processor.stop()


async def save_upload_file(upload_file: UploadFile) -> str:
    """异步保存上传的文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)

    return file_path


@app.post("/animate", response_model=TaskStatus)
async def create_animation(
        ref_image: UploadFile = File(...),
        drive_video: UploadFile = File(...),
        params: AnimationRequest = Form(...)
):
    """创建新的动画生成任务"""
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # 异步保存上传的文件
        ref_image_path = await save_upload_file(ref_image)
        drive_video_path = await save_upload_file(drive_video)

        # 创建任务记录
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending"
        }

        # 将任务添加到处理队列
        task_processor.add_task(
            task_id,
            ref_image_path,
            drive_video_path,
            params.dict()
        )

        return TaskStatus(
            task_id=task_id,
            status="pending"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatus(**tasks[task_id])


@app.get("/output/{task_id}")
async def get_output_video(task_id: str):
    """获取输出视频"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    if not os.path.exists(task["output_path"]):
        raise HTTPException(status_code=404, detail="Output file not found")

    return {"file_path": task["output_path"]}


if __name__ == "__main__":
    args = parse_args()
    create_startup_event(args.token)  # 传递 token
    uvicorn.run(
        "api:app",  # 确保这里使用的是你的文件名，比如 main.py
        host=args.host,
        port=args.port,
        reload=args.debug
    )
