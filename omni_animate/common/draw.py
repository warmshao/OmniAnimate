import math
import pdb

import numpy as np
import matplotlib
import cv2

eps = 0.01


def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]


def draw_bodypose(canvas, candidate, subset, score, score_thred=0.4):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < score_thred or conf[1] < score_thred:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas


def draw_body_foot_pose(canvas, candidate, subset, score, score_thred=0.4):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [14, 19], [11, 20]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [170, 255, 255], [255, 255, 0]]

    for i in range(19):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < score_thred or conf[1] < score_thred:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(20):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, all_scores):
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas


def draw_pose(pose, H, W, ref_w=720, **kwargs):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    if kwargs.get("draw_foot", True):
        canvas = draw_body_foot_pose(canvas, candidate, subset, score=bodies['score'])
    else:
        canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'])

    ########################################### draw hand pose #####################################################
    if kwargs.get("draw_hand", True):
        canvas = draw_handpose(canvas, hands, pose['hands_score'])

    ########################################### draw face pose #####################################################
    if kwargs.get("draw_face", True):
        canvas = draw_facepose(canvas, faces, pose['faces_score'])

    canvas = cv2.resize(canvas, (W, H))
    if kwargs.get("to_rgb", False):
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def draw_pose_v2(keypoints2d, H, W, ref_w=720, **kwargs):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    score_thred = kwargs.get("score_thred", 0.3)
    keypoints_info = keypoints2d.copy()
    foot_kpts2d = keypoints2d[:, [17, 20]].copy()
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

    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    if kwargs.get("draw_foot", True):
        canvas = draw_body_foot_pose(canvas, candidate, subset, score=bodies['score'], score_thred=score_thred)
    else:
        canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'], score_thred=score_thred)

    ########################################### draw hand pose #####################################################
    if kwargs.get("draw_hand", True):
        canvas = draw_handpose(canvas, hands, pose['hands_score'])

    ########################################### draw face pose #####################################################
    if kwargs.get("draw_face", True):
        canvas = draw_facepose(canvas, faces, pose['faces_score'])
    canvas = cv2.resize(canvas, (W, H))
    if kwargs.get("to_rgb", False):
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas
