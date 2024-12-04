import pdb

import cv2
import numpy as np
from torch.cuda import nvtx
import torch

from .base_model import BaseModel
from .predictor import numpy_to_torch_dtype_dict


class YoloHumanDetectModel(BaseModel):
    """
    人体检测的网络
    """

    def __init__(self, **kwargs):
        super(YoloHumanDetectModel, self).__init__(**kwargs)
        self.det_thred = kwargs.get("det_thred", 0.4)
        self.nms_thred = kwargs.get("nms_thred", 0.5)

    def input_process(self, *data, **kwargs):
        INPUT_H, INPUT_W = self.input_shapes[0][1][2:4]
        image = data[0]
        h, w = image.shape[:2]
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        scale = min(r_w, r_h)
        tw = int(scale * w)
        th = int(scale * h)
        tx1 = 0
        ty1 = 0
        tx2 = INPUT_W - tw
        ty2 = INPUT_H - th

        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        image = image.astype(np.float32) / 255
        image = np.transpose(image, (2, 0, 1))
        return image[None], scale

    def output_process(self, *data, **kwargs):
        preds, scale = data
        preds = preds[0][0]
        boxes = preds[:, :4]
        scores = preds[:, 4]
        labels = preds[:, 5]
        labels = labels[scores > self.det_thred]
        boxes = boxes[scores > self.det_thred]
        scores = scores[scores > self.det_thred]

        boxes = boxes[labels == 0]
        scores = scores[labels == 0]
        if len(boxes):
            boxes /= scale
            return boxes
        else:
            return None

    def predict_trt(self, *data, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i].to(device=self.device, dtype=numpy_to_torch_dtype_dict[inp['dtype']])
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
        image, scale = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(image)
        else:
            preds = self.predictor.predict(image, **kwargs)
        outputs = self.output_process(preds, scale, **kwargs)
        return outputs
