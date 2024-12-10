# -*- coding: utf-8 -*-
# @Time    : 2024/12/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : OmniAnimate
# @FileName: test_facefusion.py
import sys
sys.path.append(".")
sys.path.append("..")

import os
import pdb
import subprocess
import json

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_facefusion():
    FACEFUSION_DIR = os.path.join(PROJECT_DIR, "third_party/facefusion")
    CUR_DIR = os.getcwd()
    os.chdir(FACEFUSION_DIR)
    job_dir = '.jobs'
    os.makedirs(os.path.join(job_dir, 'queued'), exist_ok=True)
    template_json = os.path.join(PROJECT_DIR, "assets/facefusion_templates/omni_animate_v1.json")
    with open(template_json, "r") as fin:
        template_data = json.load(fin)
    image_path = os.path.join(PROJECT_DIR, "assets/examples/img.png")
    video_path = os.path.join(PROJECT_DIR, "assets/examples/001.mp4")
    output_path = os.path.join(PROJECT_DIR, "results/001-img.mp4")
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
