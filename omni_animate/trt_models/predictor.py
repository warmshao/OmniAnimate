import pdb
import threading
import os
import time

import numpy as np
import onnxruntime

import torch
from torch.cuda import nvtx
from collections import OrderedDict

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class OnnxRuntimePredictor:
    """
    OnnxRuntime Prediction
    """

    def __init__(self, **kwargs):
        model_path = kwargs.get("model_path", "")  # 用模型路径区分是否是一样的实例
        assert os.path.exists(model_path), "model path must exist!"
        # print("loading ort model:{}".format(model_path))
        self.debug = kwargs.get("debug", False)
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']

        print(f"OnnxRuntime use {providers}")
        opts = onnxruntime.SessionOptions()
        # opts.inter_op_num_threads = kwargs.get("num_threads", 4)
        # opts.intra_op_num_threads = kwargs.get("num_threads", 4)
        # opts.log_severity_level = 3
        self.onnx_model = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=opts)
        self.inputs = self.onnx_model.get_inputs()
        self.outputs = self.onnx_model.get_outputs()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o.name, o.shape, o.type))
            if self.debug:
                print(f"ort {i} -> {o.name} -> {o.shape}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o.name, o.shape, o.type))
            if self.debug:
                print(f"ort output {i} -> {o.name} -> {o.shape}")
        return specs

    def predict(self, *data):
        input_feeds = {}
        for i in range(len(data)):
            if self.inputs[i].type == 'tensor(float16)':
                input_feeds[self.inputs[i].name] = data[i].astype(np.float16)
            else:
                input_feeds[self.inputs[i].name] = data[i].astype(np.float32)
        results = self.onnx_model.run(None, input_feeds)
        return results

    def __del__(self):
        del self.onnx_model
        self.onnx_model = None


class OnnxRuntimePredictorSingleton(OnnxRuntimePredictor):
    """
    单例模式，防止模型被加载多次
    """
    _instance_lock = threading.Lock()
    _instance = {}

    def __new__(cls, *args, **kwargs):
        model_path = kwargs.get("model_path", "")  # 用模型路径区分是否是一样的实例
        assert os.path.exists(model_path), "model path must exist!"
        # 单例模式，避免重复加载模型
        with OnnxRuntimePredictorSingleton._instance_lock:
            if model_path not in OnnxRuntimePredictorSingleton._instance or \
                    OnnxRuntimePredictorSingleton._instance[model_path].onnx_model is None:
                OnnxRuntimePredictorSingleton._instance[model_path] = OnnxRuntimePredictor(**kwargs)

        return OnnxRuntimePredictorSingleton._instance[model_path]


def get_predictor(**kwargs):
    predict_type = kwargs.get("predict_type", "trt")
    if predict_type == "ort":
        return OnnxRuntimePredictorSingleton(**kwargs)
    else:
        raise NotImplementedError
