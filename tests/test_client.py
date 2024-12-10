"""
python tests/test_client.py \
--token omni-d99745d8-b2e6-4e2a-9982-9f930fca53e0 \
--ref-image assets/examples/img.png \
--drive-video assets/examples/001.mp4
"""
import requests
import time
import argparse
import os
from typing import Optional
from pathlib import Path
from urllib.parse import urlencode
import time


class OmniAnimateClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def create_animation(
            self,
            ref_image_path: str,
            drive_video_path: str,
            token: str,
            height: int = 1024,
            width: int = 576,
            keep_ratio: bool = True,
            keep_ref_dim: bool = False,
            stride: int = 1,
            steps: int = 20,
            seed: int = 1234,
            guidance_scale: float = 3.0,
            wait_complete: bool = True,
            check_interval: float = 10.0
    ) -> Optional[str]:
        """
        创建并处理动画任务

        Args:
            ref_image_path: 参考图片路径
            drive_video_path: 驱动视频路径
            token: API token
            height: 输出高度
            width: 输出宽度
            keep_ratio: 是否保持宽高比
            keep_ref_dim: 是否保持参考图像尺寸
            stride: 步长
            seed: 随机种子
            guidance_scale: 引导尺度
            wait_complete: 是否等待任务完成
            check_interval: 检查任务状态的间隔（秒）

        Returns:
            如果wait_complete为True，返回输出视频路径；否则返回任务ID
        """
        # 检查文件是否存在
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"参考图片不存在: {ref_image_path}")
        if not os.path.exists(drive_video_path):
            raise FileNotFoundError(f"驱动视频不存在: {drive_video_path}")

        # Prepare files and form data
        files = {
            'ref_image': (os.path.basename(ref_image_path), open(ref_image_path, 'rb'), 'image/jpeg'),
            'drive_video': (os.path.basename(drive_video_path), open(drive_video_path, 'rb'), 'video/mp4')
        }

        # Prepare form data
        params = {
            'token': token,
            'height': height,
            'width': width,
            'keep_ratio': keep_ratio,
            'keep_ref_dim': keep_ref_dim,
            'stride': stride,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'num_inference_steps': steps
        }
        print(params)

        try:
            # 发送请求
            import json
            response = requests.post(
                f"{self.base_url}/animate",
                files=files,
                data=params
            )
            response.raise_for_status()

            task_data = response.json()
            task_id = task_data['task_id']

            if not wait_complete:
                return task_id

            # 等待任务完成
            while True:
                status = self.get_task_status(task_id)
                if status['status'] == 'completed':
                    return self.get_output_video(task_id)
                elif status['status'] == 'failed':
                    raise RuntimeError(f"任务失败: {status.get('error', '未知错误')}")

                time.sleep(check_interval)

        finally:
            # 关闭文件
            for file in files.values():
                file[1].close()

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态"""
        response = requests.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()

    def get_output_video(self, task_id: str, save_dir: str = "./results") -> str:
        """获取输出视频路径"""
        os.makedirs(save_dir, exist_ok=True)
        response = requests.get(f"{self.base_url}/output/{task_id}", stream=True)
        # 检查响应状态码
        if response.status_code == 200:
            # 从响应头获取文件名
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                filename = content_disposition.split('filename=')[-1].strip('"')
            else:
                filename = f"output_video-{time.time()}.mp4"  # 默认文件名

            # 打开一个文件用于写入二进制数据
            with open(os.path.join(save_dir, filename), "wb") as f:
                # 分块写入文件
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉空块
                        f.write(chunk)
            print(f"视频已成功保存为 {os.path.join(save_dir, filename)}")
        else:
            print(f"请求失败，状态码: {response.status_code}")


def main():
    parser = argparse.ArgumentParser(description='OmniAnimate API 客户端')
    parser.add_argument('--ref-image', required=True, help='参考图片路径')
    parser.add_argument('--drive-video', required=True, help='驱动视频路径')
    parser.add_argument('--token', required=True, help='API token')
    parser.add_argument('--server', default='http://localhost:6006', help='服务器地址')
    parser.add_argument('--height', type=int, default=1024, help='输出高度')
    parser.add_argument('--width', type=int, default=576, help='输出宽度')
    parser.add_argument('--no-keep-ratio', action='store_false', dest='keep_ratio',
                        help='不保持宽高比')
    parser.add_argument('--keep-ref-dim', action='store_true',
                        help='保持参考图像尺寸')
    parser.add_argument('--stride', type=int, default=1, help='步长')
    parser.add_argument('--denoise_steps', type=int, default=20, help='denoise_steps')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--guidance-scale', type=float, default=3.0,
                        help='引导尺度')
    parser.add_argument('--async', action='store_true', dest='async_mode',
                        help='异步模式（不等待完成）')

    args = parser.parse_args()

    client = OmniAnimateClient(args.server)

    try:
        result = client.create_animation(
            ref_image_path=args.ref_image,
            drive_video_path=args.drive_video,
            token=args.token,
            height=args.height,
            width=args.width,
            keep_ratio=args.keep_ratio,
            keep_ref_dim=args.keep_ref_dim,
            stride=args.stride,
            steps=args.denoise_steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            wait_complete=not args.async_mode
        )

        if args.async_mode:
            print(f"任务已提交，任务ID: {result}")

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
