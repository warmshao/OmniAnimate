# -*- coding: utf-8 -*-
# @Time    : 2024/12/5
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : OmniAnimate
# @FileName: test_models.py
import sys

sys.path.append(".")
sys.path.append("..")

import os
import pdb
import cv2
import time
import numpy as np
from datetime import datetime


def test_yolo_human_detect_model():
    """
    测试 YoloHumanDetectModel
    Returns:

    """
    from omni_animate.trt_models.yolo_human_detect_model import YoloHumanDetectModel

    det_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/preprocess/yolov10x.onnx",
    )

    det_model = YoloHumanDetectModel(**det_kwargs)

    img_path = "assets/examples/img.png"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(20):
        t0 = time.time()
        trt_rets = det_model.predict(img_rgb)
        print(time.time() - t0)

    date_str = datetime.now().strftime("%m-%d-%H-%M")
    result_dir = "./results/{}-{}".format(YoloHumanDetectModel.__name__, date_str)
    os.makedirs(result_dir, exist_ok=True)

    for i, box in enumerate(trt_rets):
        img_bgr = cv2.rectangle(img_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
    print(os.path.join(result_dir, os.path.basename(img_path)))
    cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), img_bgr)


def test_rtmw_bodypose2d_model():
    from omni_animate.trt_models.rtmw_body_pose2d_model import RTMWBodyPose2dModel
    from omni_animate.trt_models.yolo_human_detect_model import YoloHumanDetectModel
    from omni_animate.common import utils
    from omni_animate.common import draw

    det_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/preprocess/yolov10x.onnx",
    )

    det_model = YoloHumanDetectModel(**det_kwargs)

    # tensorrt 模型加载
    pose_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/preprocess/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.onnx",
    )
    pose_model = RTMWBodyPose2dModel(**pose_kwargs)
    img_path = "assets/examples/img.png"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    bbox = det_model.predict(img_rgb)
    bbox = bbox.tolist()[0]
    bbox = utils.xyxy2xywh(bbox)
    for _ in range(20):
        t0 = time.time()
        keypoints = pose_model.predict(img_rgb, bbox)
        print(time.time() - t0)

    date_str = datetime.now().strftime("%m-%d-%H-%M")
    result_dir = "./results/{}-{}".format(RTMWBodyPose2dModel.__name__, date_str)
    os.makedirs(result_dir, exist_ok=True)
    img_draw = draw.draw_pose_v2(keypoints.astype(np.float32), H, W, draw_foot=True)
    cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), img_draw)
    print(os.path.join(result_dir, os.path.basename(img_path)))


if __name__ == '__main__':
    # test_yolo_human_detect_model()
    test_rtmw_bodypose2d_model()
