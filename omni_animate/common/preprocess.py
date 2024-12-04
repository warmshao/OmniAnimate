# -*- coding: utf-8 -*-
# @Time    : 2024/9/7 11:07
# @Project : AnimateMaster
# @FileName: preprocess.py
import pdb

import cv2
import numpy as np
from tqdm import tqdm


def convert_openpose_to_dict(keypoints, H, W, score_thred=0.3):
    keypoints_info = keypoints.copy()
    foot_kpts2d = keypoints[:, [17, 20]].copy()
    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    # neck score when visualizing pred
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > score_thred,
        keypoints_info[:, 6, 2:4] > score_thred).astype(int)
    new_keypoints_info = np.insert(
        keypoints_info, 17, neck, axis=1)
    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
    new_keypoints_info[:, [18, 19]] = foot_kpts2d.copy()
    keypoints_info = new_keypoints_info
    candidate, score = keypoints_info[..., :2], keypoints_info[..., 2]
    nums, _, locs = candidate.shape
    candidate[..., 0] /= float(W)
    candidate[..., 1] /= float(H)
    body = candidate[:, :20].copy()
    body = body.reshape(nums * 20, locs)
    subset = score[:, :20].copy()
    for i in range(len(subset)):
        for j in range(len(subset[i])):
            if subset[i][j] > score_thred:
                subset[i][j] = int(20 * i + j)
            else:
                subset[i][j] = -1

    faces = candidate[:, 24:92]

    hands = candidate[:, 92:113]
    hands = np.vstack([hands, candidate[:, 113:]])

    faces_score = score[:, 24:92]
    hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

    bodies = dict(candidate=body, subset=subset, score=score[:, :20])
    pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)
    return pose


def preprocess_openpose(src_vpath, ref_image_path, **kwargs):
    from animate_master.trt_models.rtmw_body_pose2d_model import RTMWBodyPose2dModel
    from animate_master.trt_models.yolo_human_detect_model import YoloHumanDetectModel
    from animate_master.common import utils
    from animate_master.common import draw
    from PIL import Image

    if kwargs.get("det_model", None) is None:
        det_kwargs = dict(
            predict_type="trt",
            model_path="./checkpoints/AnimateMaster/yolov10x.trt",
        )

        det_model = YoloHumanDetectModel(**det_kwargs)
    else:
        det_model = kwargs.get("det_model")

    if kwargs.get("pose_model", None) is None:
        # tensorrt 模型加载
        pose_kwargs = dict(
            predict_type="trt",
            model_path="./checkpoints/AnimateMaster/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.trt",
        )
        pose_model = RTMWBodyPose2dModel(**pose_kwargs)
    else:
        pose_model = kwargs.get("pose_model")
    score_thred = kwargs.get("score_thred", 0.3)

    # select ref-keypoint from reference pose for pose rescale
    ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
    ref_H, ref_W = ref_image.shape[:2]
    bbox = det_model.predict(ref_image)
    if bbox is None:
        bbox = [0, 0, ref_W, ref_H]
    else:
        bbox = bbox.tolist()[0]
    bbox = utils.xyxy2xywh(bbox)
    keypoints_ref = pose_model.predict(ref_image, bbox)
    keypoints_ref = keypoints_ref.astype(np.float32)
    ref_pose = convert_openpose_to_dict(keypoints_ref, ref_H, ref_W, score_thred)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
                       if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    detected_poses = []
    vcap = cv2.VideoCapture(src_vpath)
    frame_num = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_num), total=frame_num, desc="pose estimation"):
        ret, frame = vcap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        bbox = det_model.predict(img_rgb)
        if bbox is None:
            bbox = [0, 0, W, H]
        else:
            bbox = bbox.tolist()[0]
        bbox = utils.xyxy2xywh(bbox)
        keypoints_ = pose_model.predict(img_rgb, bbox)
        keypoints_ = keypoints_.astype(np.float32)
        detected_poses.append(convert_openpose_to_dict(keypoints_, H, W, score_thred))

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 20])[:,
                      ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw = H, W
    ax = ay / (fh / fw / ref_H * ref_W)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw.draw_pose(detected_pose, ref_H, ref_W, **kwargs)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def pose_align(kpts_src, kpts_ref):
    kpts_align = kpts_src.copy()
    # 先做整体的缩放
    src_height = max((kpts_align[0, 15, 1] + kpts_align[0, 16, 1]) / 2 - kpts_align[0, 0, 1], 0)
    ref_height = max((kpts_ref[15, 1] + kpts_ref[16, 1]) / 2 - kpts_ref[0, 1], 0)
    scale_h = ref_height / src_height
    print(f"scale_h:{scale_h}")
    kpts_align[:, :, 1] *= scale_h

    src_width = np.linalg.norm(kpts_align[0, 11, :2] - kpts_align[0, 12, :2])
    ref_width = np.linalg.norm(kpts_ref[11, :2] - kpts_ref[12, :2])
    scale_w = ref_width / src_width
    print(f"scale_w:{scale_w}")
    kpts_align[:, :, 0] *= scale_w

    ref_mid = (kpts_ref[11, :2] + kpts_ref[12, :2]) / 2
    src_mids = (kpts_align[:, 11, :2] + kpts_align[:, 12, :2]) / 2

    kpts_align[:, :, :2] = kpts_align[:, :, :2] - src_mids[:, None] + (src_mids[:] - src_mids[:1])[:,
                                                                      None] * 0.8 + \
                           ref_mid[None, None]
    src_mid = (kpts_align[0, 11, :2] + kpts_align[0, 12, :2]) / 2

    # 针对下半身缩放
    src_leg_len = (np.linalg.norm(kpts_align[0, 15, :2] - kpts_align[0, 11, :2]) + np.linalg.norm(
        kpts_align[0, 16, :2] - kpts_align[0, 12, :2])) / 2
    ref_leg_len = (np.linalg.norm(kpts_ref[15, :2] - kpts_ref[11, :2]) + np.linalg.norm(
        kpts_ref[16, :2] - kpts_ref[12, :2])) / 2
    leg_scale = ref_leg_len / src_leg_len
    print(f"leg_scale:{leg_scale}")
    kpts_align[:, list(range(13, 23)), 1] = (kpts_align[:, list(range(13, 23)), 1] -
                                             src_mid[None, None, 1]) * leg_scale + src_mid[
                                                None, None, 1]

    # 针对上半身缩放
    src_upper_h = (kpts_align[0, 5, 1] - kpts_align[0, 11, 1] + kpts_align[0, 6, 1] - kpts_align[
        0, 12, 1]) / 2
    ref_upper_h = (kpts_ref[5, 1] - kpts_ref[11, 1] + kpts_ref[6, 1] - kpts_ref[12, 1]) / 2
    upper_h_scale = ref_upper_h / src_upper_h
    print(f"upper_h_scale:{upper_h_scale}")
    kpts_align[:, list(range(11)) + list(range(91, 133)), 1] = (kpts_align[:, list(range(11)) + list(range(91, 133)),
                                                                1] -
                                                                src_mid[None, None, 1]) * upper_h_scale + src_mid[
                                                                   None, None, 1]

    src_upper_w = np.linalg.norm(kpts_align[0, 5, :2] - kpts_align[0, 6, :2])
    ref_upper_w = np.linalg.norm(kpts_ref[5, :2] - kpts_ref[6, :2])
    upper_w_scale = ref_upper_w / src_upper_w
    print(f"upper_w_scale:{upper_w_scale}")
    kpts_align[:, list(range(5, 11)) + list(range(91, 133)), 0] = (kpts_align[:,
                                                                   list(range(5, 11)) + list(range(91, 133)),
                                                                   0] -
                                                                   src_mid[None, None, 0]) * upper_w_scale + src_mid[
                                                                      None, None, 0]

    # 对手臂做缩放
    hand_scale = 0.8
    kpts_align[:, [7, 9] + list(range(91, 112)), :2] = (kpts_align[:, [7, 9] + list(range(91, 112)), :2] -
                                                        kpts_align[:, [5], :2]) * hand_scale + kpts_align[:, [5], :2]
    kpts_align[:, [8, 10] + list(range(112, 133)), :2] = (kpts_align[:, [8, 10] + list(range(112, 133)), :2] -
                                                          kpts_align[:, [6], :2]) * hand_scale + kpts_align[:, [6], :2]

    # 针对头部缩放
    head_mid = kpts_align[:, 0, :2]
    src_head_w = np.linalg.norm(kpts_align[0, 3, :2] - kpts_align[0, 4, :2])
    ref_head_w = np.linalg.norm(kpts_ref[3, :2] - kpts_ref[4, :2])
    head_w_scale = ref_head_w / src_head_w
    print(f"head_w_scale:{head_w_scale}")
    kpts_align[:, list(range(5)) + list(range(23, 91)), 0] = (kpts_align[:, list(range(5)) + list(range(23, 91)), 0] -
                                                              head_mid[:, None, 0]) * head_w_scale + head_mid[
                                                                                                     :, None, 0]

    head_mid = (kpts_align[:, 5, :2] + kpts_align[:, 6, :2]) / 2.0
    src_head_h = np.linalg.norm(kpts_align[0, 0, :2] - head_mid)
    ref_head_h = np.linalg.norm(kpts_ref[0, :2] - head_mid)
    head_h_scale = ref_head_h / src_head_h
    print(f"head_h_scale:{head_h_scale}")
    kpts_align[:, list(range(5)) + list(range(23, 91)), 1] = (kpts_align[:, list(range(5)) + list(range(23, 91)), 1] -
                                                              head_mid[:, None, 1]) * head_h_scale + head_mid[
                                                                                                     :, None, 1]

    return kpts_align


def preprocess_openpose_v2(src_vpath, ref_image_path, **kwargs):
    from animate_master.trt_models.rtmw_body_pose2d_model import RTMWBodyPose2dModel
    from animate_master.trt_models.yolo_human_detect_model import YoloHumanDetectModel
    from animate_master.common import utils
    from animate_master.common import draw
    from PIL import Image

    if kwargs.get("det_model", None) is None:
        det_kwargs = dict(
            predict_type="trt",
            model_path="./checkpoints/AnimateMaster/yolov10x.trt",
        )

        det_model = YoloHumanDetectModel(**det_kwargs)
    else:
        det_model = kwargs.get("det_model")

    if kwargs.get("pose_model", None) is None:
        # tensorrt 模型加载
        pose_kwargs = dict(
            predict_type="trt",
            model_path="./checkpoints/AnimateMaster/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.trt",
        )
        pose_model = RTMWBodyPose2dModel(**pose_kwargs)
    else:
        pose_model = kwargs.get("pose_model")
    score_thred = kwargs.get("score_thred", 0.3)

    # select ref-keypoint from reference pose for pose rescale
    ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
    ref_H, ref_W = ref_image.shape[:2]
    bbox = det_model.predict(ref_image)
    if bbox is None:
        bbox = [0, 0, ref_W, ref_H]
    else:
        bbox = bbox.tolist()[0]
    bbox = utils.xyxy2xywh(bbox)
    keypoints_ref = pose_model.predict(ref_image, bbox)
    keypoints_ref = keypoints_ref.astype(np.float32)

    detected_poses = []
    vcap = cv2.VideoCapture(src_vpath)
    frame_num = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_num), total=frame_num, desc="pose estimation"):
        ret, frame = vcap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        bbox = det_model.predict(img_rgb)
        if bbox is None:
            bbox = [0, 0, W, H]
        else:
            bbox = bbox.tolist()[0]
        bbox = utils.xyxy2xywh(bbox)
        keypoints_ = pose_model.predict(img_rgb, bbox)
        keypoints_ = keypoints_.astype(np.float32)
        detected_poses.append(keypoints_)
    detected_poses = np.concatenate(detected_poses)
    detected_poses = pose_align(detected_poses, keypoints_ref[0].copy())

    output_pose = []
    # pose rescale
    for detected_pose in detected_poses:
        detected_pose_ = convert_openpose_to_dict(detected_pose[None], ref_H, ref_W, score_thred)
        im = draw.draw_pose(detected_pose_, ref_H, ref_W, **kwargs)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def preprocess_openpose_image(ref_image_path, **kwargs):
    from ..trt_models.rtmw_body_pose2d_model import RTMWBodyPose2dModel
    from ..trt_models.yolo_human_detect_model import YoloHumanDetectModel
    from ..common import utils
    from ..common import draw
    from PIL import Image

    if kwargs.get("det_model", None) is None:
        det_kwargs = dict(
            predict_type="trt",
            model_path="./checkpoints/AnimateMaster/yolov10x.trt",
        )

        det_model = YoloHumanDetectModel(**det_kwargs)
    else:
        det_model = kwargs.get("det_model")

    if kwargs.get("pose_model", None) is None:
        # tensorrt 模型加载
        pose_kwargs = dict(
            predict_type="trt",
            model_path="./checkpoints/AnimateMaster/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.trt",
        )
        pose_model = RTMWBodyPose2dModel(**pose_kwargs)
    else:
        pose_model = kwargs.get("pose_model")

    score_thred = kwargs.get("score_thred", 0.3)

    # select ref-keypoint from reference pose for pose rescale
    ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
    ref_H, ref_W = ref_image.shape[:2]
    bbox = det_model.predict(ref_image)
    if bbox is None:
        bbox = [0, 0, ref_W, ref_H]
    else:
        bbox = bbox.tolist()[0]
    bbox = utils.xyxy2xywh(bbox)
    keypoints_ref = pose_model.predict(ref_image, bbox)
    keypoints_ref = keypoints_ref.astype(np.float32)
    ref_pose = convert_openpose_to_dict(keypoints_ref, ref_H, ref_W, score_thred)
    # pose rescale
    ref_image_draw = draw.draw_pose(ref_pose, ref_H, ref_W, **kwargs)
    return ref_image_draw
