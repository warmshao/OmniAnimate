# -*- coding: utf-8 -*-
# @Time    : 2024/12/8
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : OmniAnimate
# @FileName: test_pipelines.py
import pdb
import sys

sys.path.append(".")


def test_animate_master_pipeline():
    import torch
    from omni_animate.pipelines.pipeline_mimic_motion import AnimateMasterPipeline
    ref_image_path = "assets/examples/img.png"
    src_video_path = "assets/examples/001.mp4"

    token = "omni-d99745d8-b2e6-4e2a-9982-9f930fca53e0"
    pipe = AnimateMasterPipeline(token=token)

    animate_vpath = pipe(ref_image_path, src_video_path, tile_size=32, tile_overlap=6,
                         height=1024, width=576, stride=1, fps=8,
                         noise_aug_strength=0, num_inference_steps=25,
                         seed=1234, min_guidance_scale=3,
                         max_guidance_scale=3, decode_chunk_size=8, token=token
                         )
    pdb.set_trace()


if __name__ == '__main__':
    test_animate_master_pipeline()
