"""
export OMNI_TOKEN=omni-ae342931-814d-4a49-97a6-cedf6af3dd18
uvicorn api:app --host 0.0.0.0 --port 6006
"""
import pdb
from typing import Annotated
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
from fastapi.responses import FileResponse

from sympy.physics.vector.printing import params

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
    guidance_scale: float = 3.0
    num_inference_steps: int = 20


class TaskStatus(BaseModel):
    task_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None


# 任务处理器
class TaskProcessor:
    def __init__(self, pipe):
        self.task_queue = multiprocessing.Queue()
        self.results = {}
        self.running = True
        self.pipe = pipe

    def start(self):
        worker = threading.Thread(target=self._worker_process)
        worker.start()
        self.result_thread = threading.Thread(target=self._collect_results)
        self.result_thread.start()

    def stop(self):
        self.running = False
        self.task_queue.put(None)  # Signal the worker to stop
        self.result_thread.join()

    def _worker_process(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break

            task_id, ref_image_path, drive_video_path, params = task
            try:
                output_path = self.pipe(
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


@app.on_event("startup")
async def startup_event():
    global task_processor
    token = os.getenv("OMNI_TOKEN", '')
    print(f"Initializing pipeline with token: {token}")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe = AnimateMasterPipeline(token=token)
    task_processor = TaskProcessor(pipe=pipe)
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
        ref_image: Annotated[UploadFile, File()],
        drive_video: Annotated[UploadFile, File()],
        token: Annotated[str, Form()],
        height: Annotated[int, Form()],
        width: Annotated[int, Form()],
        keep_ratio: Annotated[bool, Form()],
        keep_ref_dim: Annotated[bool, Form()],
        stride: Annotated[int, Form()],
        seed: Annotated[int, Form()],
        guidance_scale: Annotated[float, Form()],
        num_inference_steps: Annotated[int, Form()]
):
    """创建新的动画生成任务"""
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        params = AnimationRequest(
            token=token,
            height=height,
            width=width,
            keep_ratio=keep_ratio,
            keep_ref_dim=keep_ref_dim,
            stride=stride,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        print(params)

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

    return FileResponse(task["output_path"], media_type='video/mp4', filename=os.path.basename(task["output_path"]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=6006, reload=True)
