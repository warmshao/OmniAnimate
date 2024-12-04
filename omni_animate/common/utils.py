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


def get_hand_box_from_kpts(kps_in, h, w, player_box, thred=0.3, min_counts=10, scale=3):
    """
    从检测到的关键点获取得到bounding box
    Args:
        kps_in: 关键点坐标
        h: 图片高
        w: 图片宽

    Returns:

    """
    if kps_in is None or player_box is None:
        return None
    px1, py1, px2, py2 = player_box
    kps = kps_in.copy()
    kps = kps[kps[:, 2] > thred]
    if len(kps) < min_counts:
        return None
    xmin = np.min(kps[:, 0])
    xmax = np.max(kps[:, 0])
    ymin = np.min(kps[:, 1])
    ymax = np.max(kps[:, 1])
    xmin = max(xmin - (xmax - xmin) // scale, 0)
    xmax = min(xmax + (xmax - xmin) // scale, w)
    ymin = max(ymin - (ymax - ymin) // scale, 0)
    ymax = min(ymax + (ymax - ymin) // scale, h)
    # # 如果面积太小，也舍弃掉
    if box_area([xmin, ymin, xmax, ymax]) < ((py2 - py1) * (px2 - px1) // 200):
        return None
    l = max(xmax - xmin, ymax - ymin)
    ll = lr = max((l - (xmax - xmin)) // 2, 0)
    lt = lb = max((l - (ymax - ymin)) // 2, 0)
    xmin = max(xmin - ll, 0)
    xmax = min(xmax + lr, w)
    ymin = max(ymin - lt, 0)
    ymax = min(ymax + lb, h)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def get_hand_boxes_from_poses(pose2d_cur, h, w, player_box, kpt_thred=0.5, iou_thred=0.5, min_hand_pnum=16):
    """
    从full body pose获取 hand box的位置
    """
    left_hand_poses = pose2d_cur[91:112].copy()
    left_score = (left_hand_poses[:, 2] > kpt_thred).sum().tolist()

    if left_score < min_hand_pnum:
        left_hand_box = None
    else:
        left_hand_box = get_hand_box_from_kpts(left_hand_poses, h, w,
                                               player_box=player_box,
                                               thred=kpt_thred,
                                               min_counts=min_hand_pnum)
        if left_hand_box is not None and \
                max(left_hand_box[2] - left_hand_box[0], left_hand_box[3] - left_hand_box[1]) < max(
            player_box[2] - player_box[0], player_box[3] - player_box[1]) / 16:
            left_hand_box = None

    # 处理右手
    right_hand_poses = pose2d_cur[112:133].copy()
    right_score = (right_hand_poses[:, 2] > kpt_thred).sum().tolist()
    if right_score < min_hand_pnum:
        right_hand_box = None
    else:
        right_hand_box = get_hand_box_from_kpts(right_hand_poses, h, w,
                                                player_box=player_box,
                                                thred=kpt_thred,
                                                min_counts=min_hand_pnum)
        if right_hand_box is not None and \
                max(right_hand_box[2] - right_hand_box[0], right_hand_box[3] - right_hand_box[1]) < max(
            player_box[2] - player_box[0], player_box[3] - player_box[1]) / 16:
            right_hand_box = None

    if left_hand_box and right_hand_box and (iou(xyxy2xywh(left_hand_box),
                                                 xyxy2xywh(right_hand_box)) > iou_thred or
                                             is_inside(xyxy2xywh(left_hand_box), xyxy2xywh(right_hand_box))):
        left_hand_box = None
        right_hand_box = None

    return left_hand_box, right_hand_box


def get_hand_boxes_from_hand_bboxs(hand_bboxs, is_rights, keypoints, min_hand_dist=100):
    lhand_kpts2d = keypoints[91:112]
    lhand_valid = lhand_kpts2d[0, -1] > 0.5
    rhand_kpts2d = keypoints[112:133]
    rhand_valid = rhand_kpts2d[0, -1] > 0.5

    # Initialize variables to store the closest boxes and minimum distances
    lhand_box = None
    rhand_box = None
    min_lhand_dist = min_hand_dist
    min_rhand_dist = min_hand_dist

    # Iterate over each bounding box and its corresponding is_right flag
    for bbox, is_right in zip(hand_bboxs, is_rights):
        # Calculate the center of the bounding box
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        # Determine if the box is for the right hand
        if is_right:
            if rhand_valid:
                # Calculate the distance to the right hand root keypoint
                dist = np.linalg.norm(bbox_center - rhand_kpts2d[rhand_kpts2d[:, -1] > 0.5, :2].mean(axis=0))
                # Update the closest right hand box if this one is closer
                if dist < min_rhand_dist:
                    min_rhand_dist = dist
                    rhand_box = bbox
        else:
            if lhand_valid:
                # Calculate the distance to the left hand root keypoint
                dist = np.linalg.norm(bbox_center - lhand_kpts2d[lhand_kpts2d[:, -1] > 0.5, :2].mean(axis=0))
                # Update the closest left hand box if this one is closer
                if dist < min_lhand_dist:
                    min_lhand_dist = dist
                    lhand_box = bbox
    return lhand_box, rhand_box


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


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def coco2openpose65(kpts2d, kpt_thred=0.5):
    """
    coco关键点转为openpose65:body25+hand40
    Parameters
    ----------
    kpts2d

    Returns
    -------

    """
    map_index = np.array(
        [1, 1, 7, 9, 11, 6, 8, 10, 7, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4, 18, 19, 20, 21, 22, 23]) - 1
    kpts2d_openpose25 = kpts2d[:, map_index]
    kpts2d_openpose25[:, 1] = kpts2d_openpose25[:, 2] * 0.5 + kpts2d_openpose25[:, 5] * 0.5
    kpts2d_openpose25[:, 8] = kpts2d_openpose25[:, 9] * 0.5 + kpts2d_openpose25[:, 12] * 0.5
    kpts2d_openpose25 = np.concatenate([kpts2d_openpose25, kpts2d[:, 92:112], kpts2d[:, 113:133]], 1)
    kpts2d_openpose25[..., -1] = kpts2d_openpose25[..., -1] > kpt_thred
    return kpts2d_openpose25


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    try:
        out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    except ValueError:
        out[:] = 0
    return out
