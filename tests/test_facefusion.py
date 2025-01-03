# -*- coding: utf-8 -*-
# @Time    : 2024/12/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : OmniAnimate
# @FileName: test_facefusion.py

import os
import pdb
import shutil
import subprocess
import json
from huggingface_hub import hf_hub_download

def test_facefusion():
    """
    测试facefusion的效果
    :return:
    """
    from omni_animate.common import constants
    PACKAGE_DIR = constants.PACKAGE_DIR
    FACEFUSION_DIR = os.path.join(PACKAGE_DIR, "third_party/facefusion")
    CUR_DIR = os.getcwd()
    os.chdir(FACEFUSION_DIR)
    job_dir = '.jobs'
    os.makedirs(os.path.join(job_dir, 'queued'), exist_ok=True)
    template_json = os.path.join(constants.CHECKPOINT_DIR, "facefusion_templates/omni_animate_v1.json")
    if not os.path.exists(template_json):
        hf_hub_download(repo_id="warmshao/OmniAnimate",
                        subfolder="facefusion_templates",
                        filename="omni_animate_v1.json",
                        local_dir=constants.CHECKPOINT_DIR
                        )
    with open(template_json, "r") as fin:
        template_data = json.load(fin)
    image_path = os.path.join(PACKAGE_DIR, "../assets/examples/img.png")
    video_path = os.path.join(PACKAGE_DIR, "../assets/examples/001.mp4")
    output_path = os.path.join(PACKAGE_DIR, "../results/001-img-facefusion2.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    job_id = "omni_animate_v1"
    template_data['steps'][0]['args']['source_paths'] = [image_path]
    template_data['steps'][0]['args']['target_path'] = video_path
    template_data['steps'][0]['args']['output_path'] = output_path
    with open(os.path.join(job_dir, "queued", f"{job_id}.json"), "w") as fw:
        json.dump(template_data, fw)
    commands = ['python', 'facefusion.py', 'job-run', job_id, '-j', job_dir]
    print(commands)
    run_ret = subprocess.run(commands).returncode
    os.chdir(CUR_DIR)
    print(output_path)
    pdb.set_trace()


if __name__ == '__main__':
    test_facefusion()
