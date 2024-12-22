# -*- coding: utf-8 -*-
# @Time    : 2024/12/8
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : OmniAnimate
# @FileName: test_pipelines.py
import pdb
import sys

sys.path.append(".")


def test_mimic_motion_pipeline():
    from omni_animate.pipelines.pipeline_mimic_motion import MimicMotionPipeline
    ref_image_path = "assets/examples/img.png"
    src_video_path = "assets/examples/001.mp4"

    pipe = MimicMotionPipeline()

    animate_vpath = pipe(ref_image_path, src_video_path, tile_size=72, tile_overlap=6,
                         height=1024, width=576, stride=2, fps=7,
                         noise_aug_strength=0, num_inference_steps=25,
                         seed=1234, min_guidance_scale=2,
                         max_guidance_scale=2, decode_chunk_size=8, use_faceswap=True
                         )
    pdb.set_trace()


if __name__ == '__main__':
    test_mimic_motion_pipeline()
