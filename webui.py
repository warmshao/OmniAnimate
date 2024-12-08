# -*- coding: utf-8 -*-
# @Time    : 2024/12/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : OmniAnimate
# @FileName: webui.py

import gradio as gr
import torch
import argparse
from typing import Optional
from omni_animate.pipelines.pipeline_animate_master import AnimateMasterPipeline

pipe = AnimateMasterPipeline()


def process_animation(
        ref_image: str,
        drive_video: str,
        token: str,
        height: int,
        width: int,
        keep_ratio: bool,
        keep_ref_dim: bool,
        stride: int,
        seed: int,
        guidance_scale: float
) -> Optional[str]:
    try:

        # Run the animation pipeline
        output_path = pipe(
            ref_image_path=ref_image,
            src_video_path=drive_video,
            tile_size=32,
            tile_overlap=6,
            height=height,
            width=width,
            stride=stride,
            fps=8,
            noise_aug_strength=0,
            num_inference_steps=25,
            keep_ratio=keep_ratio,
            keep_ref_dim=keep_ref_dim,
            seed=seed,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=guidance_scale,
            decode_chunk_size=8,
            token=token
        )

        return output_path
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description='OmniAnimate WebUI')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务器IP地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=6006,
                        help='服务器端口 (默认: 6006)')
    parser.add_argument('--token', type=str, default='',
                        help='token')
    parser.add_argument('--share', action='store_true',
                        help='是否创建公共链接 (默认: False)')
    parser.add_argument('--debug', action='store_true',
                        help='是否启用调试模式 (默认: False)')
    return parser.parse_args()


# Create the Gradio interface
def create_ui():
    with gr.Blocks(title="OmniAnimate") as demo:
        gr.Markdown("# OmniAnimate - 视频驱动图片动画")

        with gr.Row():
            # Left column - Reference image
            with gr.Column():
                gr.Markdown("### 参考图片")
                ref_image_input = gr.Image(type="filepath", label="上传参考图片")

            # Middle column - Driving video
            with gr.Column():
                gr.Markdown("### 驱动视频")
                drive_video_input = gr.Video(label="上传驱动视频")

            # Right column - Output and parameters
            with gr.Column():
                gr.Markdown("### 输出视频")
                output_video = gr.Video(label="生成的视频")

                # Parameters
                with gr.Box():
                    gr.Markdown("### 参数设置")
                    token_input = gr.Textbox(label="Token", placeholder="输入你的token")
                    height_input = gr.Number(label="Height", value=1024, precision=0)
                    width_input = gr.Number(label="Width", value=576, precision=0)
                    keep_ratio = gr.Checkbox(label="Keep Ratio", value=True)
                    keep_ref_dim = gr.Checkbox(label="Keep Reference Dimensions", value=False)
                    stride_input = gr.Number(label="Stride", value=1, precision=0)
                    seed_input = gr.Number(label="Seed", value=1234, precision=0)
                    guidance_input = gr.Number(label="Guidance Scale", value=3.0)

                    run_btn = gr.Button("运行", variant="primary")

        # Set up the processing function
        run_btn.click(
            fn=process_animation,
            inputs=[
                ref_image_input,
                drive_video_input,
                token_input,
                height_input,
                width_input,
                keep_ratio,
                keep_ref_dim,
                stride_input,
                seed_input,
                guidance_input
            ],
            outputs=output_video
        )

    return demo


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )
