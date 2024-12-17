"""
python tests/api_client.py \
--server http://129.213.81.69:6006 \
--token omni-ae342931-814d-4a49-97a6-cedf6af3dd18 \
--ref-image assets/examples/img.png \
--drive-video assets/examples/001.mp4
"""

import os
import requests
import time
import argparse
from typing import Optional


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
        Create and process an animation task.

        Args:
            ref_image_path: Path to the reference image.
            drive_video_path: Path to the driving video.
            token: API token.
            height: Output height.
            width: Output width.
            keep_ratio: Whether to maintain aspect ratio.
            keep_ref_dim: Whether to keep the reference image dimensions.
            stride: Stride.
            seed: Random seed.
            guidance_scale: Guidance scale.
            wait_complete: Whether to wait for task completion.
            check_interval: Interval to check task status (in seconds).

        Returns:
            If wait_complete is True, returns the output video path; otherwise, returns the task ID.
        """
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
        if not os.path.exists(drive_video_path):
            raise FileNotFoundError(f"Driving video not found: {drive_video_path}")

        files = {
            'ref_image': (os.path.basename(ref_image_path), open(ref_image_path, 'rb'), 'image/jpeg'),
            'drive_video': (os.path.basename(drive_video_path), open(drive_video_path, 'rb'), 'video/mp4')
        }

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

            while True:
                status = self.get_task_status(task_id)
                if status['status'] == 'completed':
                    return self.get_output_video(task_id)
                elif status['status'] == 'failed':
                    raise RuntimeError(f"Task failed: {status.get('error', 'Unknown error')}")

                time.sleep(check_interval)

        finally:
            for file in files.values():
                file[1].close()

    def get_task_status(self, task_id: str) -> dict:
        """Get task status."""
        response = requests.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()

    def get_output_video(self, task_id: str, save_dir: str = "./results") -> str:
        """Get output video path."""
        os.makedirs(save_dir, exist_ok=True)
        response = requests.get(f"{self.base_url}/output/{task_id}", stream=True)
        if response.status_code == 200:
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                filename = content_disposition.split('filename=')[-1].strip('"')
            else:
                filename = f"output_video-{time.time()}.mp4"

            with open(os.path.join(save_dir, filename), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Video successfully saved as {os.path.join(save_dir, filename)}")
        else:
            print(f"Request failed, status code: {response.status_code}")


def main():
    parser = argparse.ArgumentParser(description='OmniAnimate API Client')
    parser.add_argument('--ref-image', required=True, help='Path to the reference image')
    parser.add_argument('--drive-video', required=True, help='Path to the driving video')
    parser.add_argument('--token', required=True, help='API token')
    parser.add_argument('--server', default='http://localhost:6006', help='Server URL')
    parser.add_argument('--height', type=int, default=1024, help='Output height')
    parser.add_argument('--width', type=int, default=576, help='Output width')
    parser.add_argument('--no-keep-ratio', action='store_false', dest='keep_ratio',
                        help='Do not maintain aspect ratio')
    parser.add_argument('--keep-ref-dim', action='store_true',
                        help='Keep reference image dimensions')
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--denoise_steps', type=int, default=25, help='Denoise steps')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--guidance-scale', type=float, default=3.0,
                        help='Guidance scale')
    parser.add_argument('--async', action='store_true', dest='async_mode',
                        help='Asynchronous mode (do not wait for completion)')

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
            print(f"Task submitted, task ID: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
