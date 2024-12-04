import cv2
import numpy as np
from torch.cuda import nvtx
import torch
from .base_model import BaseModel
from .predictor import numpy_to_torch_dtype_dict
from . import utils


class RTMWBodyPose2dModel(BaseModel):

    def __init__(self, **kwargs):
        super(RTMWBodyPose2dModel, self).__init__(**kwargs)

    def input_process(self, *data, **kwargs):
        img, bbox = data
        h, w = img.shape[:2]
        INPUT_H, INPUT_W = self.input_shapes[0][1][1:3]
        input_size = [INPUT_W, INPUT_H]
        c, s = utils._box2cs(bbox, input_size, rescale=1.25)
        r = 0
        trans = utils.get_affine_transform(c, s, r, input_size)
        img = cv2.warpAffine(
            img,
            trans, (int(input_size[0]), int(input_size[1])),
            flags=cv2.INTER_LINEAR)[None, ...]
        img = img.astype(np.float32)
        img_info = {"c": c, "r": r, "s": s, "input_size": input_size, "img_dim": [h, w]}
        return img, [img_info]

    def output_process(self, *data, **kwargs):
        pose_preds, img_metas = data
        bbox_centers = img_metas[0]["c"]
        bbox_scales = img_metas[0]["s"]
        input_size = img_metas[0]['input_size']
        h, w = img_metas[0]["img_dim"]
        pose_preds[:, :, :2] = pose_preds[:, :, :2] / input_size * bbox_scales + bbox_centers - 0.5 * bbox_scales
        x_coords = pose_preds[:, :, 0]
        y_coords = pose_preds[:, :, 1]
        # 检查是否超出边界
        out_of_bounds = (x_coords < 0) | (x_coords >= w) | (y_coords < 0) | (y_coords >= h)
        pose_preds[:, :, -1] = np.where(out_of_bounds, -1, pose_preds[:, :, -1])
        return pose_preds

    def predict_trt(self, *data, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i].to(device=self.device,
                                                    dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            else:
                feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                      dtype=numpy_to_torch_dtype_dict[inp['dtype']])
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            output_shape = kwargs.get("output_shape", {})
            if out["name"] in output_shape:
                out_shape = output_shape[out["name"]]
            else:
                out_shape = out["shape"]
            out_tensor = preds_dict[out["name"]][:np.prod(out_shape)].reshape(*out_shape)
            outs.append(out_tensor.cpu().numpy())
        nvtx.range_pop()
        return outs

    def predict(self, *data, **kwargs):
        image, img_info = self.input_process(*data, **kwargs)
        if self.predict_type == "trt":
            preds = self.predict_trt(image, **kwargs)
        else:
            preds = self.predictor.predict(image)
        preds = self.output_process(preds[0], img_info, **kwargs)
        return preds
