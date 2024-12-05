# -*- coding: utf-8 -*-
# @Time    : 2024/8/18
# @Project : OmniAnimate
# @FileName: utils.py
import pdb

import numpy as np
import av
from PIL import Image
import os
import shutil
import torch
import os.path as osp


def xyxy2xywh(bbox):
    """
    坐标 xyxy2xywh
    :param bbox:
    :return:
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def box_area(bbox):
    """
    bbox的面积
    :param bbox:
    :return:
    """
    x1, y1, x2, y2 = bbox
    return (y2 - y1 + 1) * (x2 - x1 + 1)


def get_player_box_by_max_area(person_bboxs, h, w):
    """
    把最大的检测框作为玩家初始框
    Args:
        person_bboxs:所有的玩家检测框
        h:高
        w:宽

    Returns:

    """
    box_ret = None
    if person_bboxs is None:
        return None
    for b in person_bboxs:
        x1, y1, x2, y2 = b
        if box_ret is None or box_area([x1, y1, x2, y2]) > box_area(box_ret):
            box_ret = [int(x1), int(y1), int(x2), int(y2)]
    xmin, ymin, xmax, ymax = box_ret
    ## 扩大玩家区域，因为有可能预测得没那么准
    xmin = max(xmin - (xmax - xmin) // 5, 0)
    xmax = min(xmax + (xmax - xmin) // 5, w)
    ymin = max(ymin - (ymax - ymin) // 5, 0)
    ymax = min(ymax + (ymax - ymin) // 5, h)
    return [xmin, ymin, xmax, ymax]


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)

    inter = max(ymax - ymin, 0) * max(xmax - xmin, 0)
    return inter / (w1 * h1 + w2 * h2 - inter + 1e-08)


def is_inside(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)

    inter = max(ymax - ymin, 0) * max(xmax - xmin, 0)
    return inter == w1 * h1 or inter == w2 * h2


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
