import inspect
import pdb
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import einops
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import _resize_with_antialiasing, _append_dims
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import numpy as np
import os
import torch.nn.functional as F
import datetime
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2
from torchvision.io import write_video
import json
import subprocess
import requests
import rsa
import json
import base64

from ..models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from ..models.pose_net import PoseNet
from ..common import utils
from ..common import constants
from ..common import preprocess
from ..trt_models.yolo_human_detect_model import YoloHumanDetectModel
from ..trt_models.rtmw_body_pose2d_model import RTMWBodyPose2dModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs


@dataclass
class MimicMotionPipelineOutput(BaseOutput):
    r"""
    Output class for mimicmotion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class MimicMotionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K]
            (https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
        pose_net ([`PoseNet`]):
            A `` to inject pose signals into unet.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            **kwargs
    ):
        self.init_vars(**kwargs)
        self.load_models(**kwargs)

    def init_vars(self, **kwargs):
        """
        初始化一些变量
        :param kwargs:
        :return:
        """
        self.omni_animate_model_id = kwargs.get("omni_animate_model_id", "warmshao/OmniAnimate")

    def load_models(self, **kwargs):
        """
        自动下载并加载模型
        :param kwargs:
        :return:
        """
        logger.info("loading models")
        device, dtype = utils.get_optimize_device()
        logger.info(f"device: {device} dtype: {dtype}")
        HF_TOKEN = os.environ.get('HF_TOKEN', '')
        # diffusion model
        svd_base_model_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        svd_base_model_path = os.path.join(constants.CHECKPOINT_DIR, "stable-video-diffusion-img2vid-xt-1-1")
        os.makedirs(svd_base_model_path, exist_ok=True)

        if not os.path.exists(os.path.join(svd_base_model_path, "unet", "config.json")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="unet",
                            filename="config.json",
                            local_dir=svd_base_model_path,
                            token=HF_TOKEN)
        unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(svd_base_model_path, subfolder="unet"))

        if not os.path.exists(os.path.join(svd_base_model_path, "vae", "config.json")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="vae",
                            filename="config.json",
                            local_dir=svd_base_model_path)
        if not os.path.exists(os.path.join(svd_base_model_path, "vae", "diffusion_pytorch_model.fp16.safetensors")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="vae",
                            filename="diffusion_pytorch_model.fp16.safetensors",
                            local_dir=svd_base_model_path,
                            token=HF_TOKEN)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_base_model_path, subfolder="vae", variant='fp16')

        if not os.path.exists(os.path.join(svd_base_model_path, "image_encoder", "config.json")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="image_encoder",
                            filename="config.json",
                            local_dir=svd_base_model_path,
                            token=HF_TOKEN)
        if not os.path.exists(os.path.join(svd_base_model_path, "image_encoder", "model.fp16.safetensors")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="image_encoder",
                            filename="model.fp16.safetensors",
                            local_dir=svd_base_model_path,
                            token=HF_TOKEN)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            svd_base_model_path, subfolder="image_encoder", variant='fp16')

        if not os.path.exists(os.path.join(svd_base_model_path, "scheduler", "scheduler_config.json")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="scheduler",
                            filename="scheduler_config.json",
                            local_dir=svd_base_model_path,
                            token=HF_TOKEN)
        scheduler = EulerDiscreteScheduler.from_pretrained(
            svd_base_model_path, subfolder="scheduler")

        if not os.path.exists(os.path.join(svd_base_model_path, "feature_extractor", "preprocessor_config.json")):
            hf_hub_download(repo_id=svd_base_model_id, subfolder="feature_extractor",
                            filename="preprocessor_config.json",
                            local_dir=svd_base_model_path,
                            token=HF_TOKEN)
        feature_extractor = CLIPImageProcessor.from_pretrained(
            svd_base_model_path, subfolder="feature_extractor")
        # pose_net
        pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

        # MimicMotion model
        mimicmotion_model_id = "tencent/MimicMotion"
        mimicmotion_base_model_path = os.path.join(constants.CHECKPOINT_DIR, "MimicMotion")
        os.makedirs(mimicmotion_base_model_path, exist_ok=True)
        mimicmotion_model_path = os.path.join(mimicmotion_base_model_path, "MimicMotion_1-1.pth")
        if not os.path.exists(mimicmotion_model_path):
            hf_hub_download(repo_id=mimicmotion_model_id,
                            filename="MimicMotion_1-1.pth",
                            local_dir=mimicmotion_base_model_path)
        mimic_state_dict = torch.load(mimicmotion_model_path, map_location=device)
        unet_state_dict = {key[5:]: val for key, val in mimic_state_dict.items() if key.startswith("unet.")}
        unet.load_state_dict(unet_state_dict, strict=False)
        unet.eval().to(device, dtype=dtype)

        pose_net_state_dict = {key[9:]: val for key, val in mimic_state_dict.items() if key.startswith("pose_net.")}
        pose_net.load_state_dict(pose_net_state_dict, strict=True)
        pose_net.eval().to(device, dtype=dtype)
        vae.eval().to(device, dtype=dtype)
        image_encoder.eval().to(device, dtype=dtype)

        # OmniAnimate preprocess
        omni_animate_model_id = "warmshao/OmniAnimate"
        ominianimate_base_model_path = os.path.join(constants.CHECKPOINT_DIR, "OmniAnimate")
        os.makedirs(ominianimate_base_model_path, exist_ok=True)
        detect_model_path = os.path.join(ominianimate_base_model_path, "preprocess", "yolov10x.onnx")
        if not os.path.exists(detect_model_path):
            hf_hub_download(repo_id=omni_animate_model_id,
                            subfolder="preprocess",
                            filename="yolov10x.onnx",
                            local_dir=ominianimate_base_model_path
                            )
        det_kwargs = dict(
            predict_type="ort",
            model_path=detect_model_path,
        )
        self.detect_model = YoloHumanDetectModel(**det_kwargs)

        pose_model_path = os.path.join(ominianimate_base_model_path, "preprocess",
                                       "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.onnx")
        if not os.path.exists(pose_model_path):
            hf_hub_download(repo_id=omni_animate_model_id, subfolder="preprocess",
                            filename="rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.onnx",
                            local_dir=ominianimate_base_model_path)
        pose_kwargs = dict(
            predict_type="ort",
            model_path=pose_model_path,
        )
        self.pose_model = RTMWBodyPose2dModel(**pose_kwargs)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
            self,
            image: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
            self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
            scale_latents=False
    ):
        image = image.to(device=device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()
        if scale_latents:
            image_latents = image_latents * self.vae.config.scaling_factor
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, " \
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. " \
                f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(
            self,
            latents: torch.Tensor,
            num_frames: int,
            decode_chunk_size: int = 8):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
            self,
            batch_size: int,
            num_frames: int,
            num_channels_latents: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def pre_process(self, ref_image_path, src_video_path, stride=1, height=576, width=1024, **kwargs):
        """
        预处理 生成pose video
        :param ref_image_path:
        :param src_video_path:
        :param stride:
        :param height:
        :param width:
        :param kwargs:
        :return:
        """
        logger.info("preprocess ref image and driving video")
        ref_image = Image.open(ref_image_path).convert("RGB")
        org_ref_w, org_ref_h = ref_image.size
        ref_pose_image = preprocess.preprocess_openpose_image(self.detect_model, self.pose_model,
                                                              ref_image_path, draw_foot=False,
                                                              draw_hand=False, draw_face=True,
                                                              to_rgb=True, score_thred=0.3)
        ref_pose_image = Image.fromarray(ref_pose_image)

        pose_images = preprocess.preprocess_openpose(self.detect_model, self.pose_model,
                                                     src_video_path, ref_image_path,
                                                     draw_foot=False, draw_hand=True,
                                                     draw_face=True,
                                                     to_rgb=True, score_thred=0.3)
        pose_images = [Image.fromarray(image) for image in pose_images]
        w, h = ref_image.size
        if kwargs.get("keep_ratio", True):
            short_size = min(height, width)
            scale = short_size / min(w, h)
            ow = int(w * scale // 64 * 64)
            oh = int(h * scale // 64 * 64)
        else:
            ow = width
            oh = height
        ref_image = ref_image.resize((ow, oh))
        pose_images = [ref_pose_image] + pose_images[::stride]
        pose_images = np.stack([np.array(img.resize((ow, oh))) for img in pose_images])
        pose_pixels = torch.from_numpy(pose_images) / 127.5 - 1
        pose_pixels = pose_pixels.permute(0, 3, 1, 2)
        return [ref_image], pose_pixels, org_ref_w, org_ref_h, ow, oh

    def post_process(self, ref_image_path, animate_video_path, **kwargs):
        """
        使用facefusion 后处理 faceswap
        :param ref_image_path:
        :param animate_video_path:
        :param kwargs:
        :return:
        """
        logger.info("using facefusion to postprocess")
        image_path = os.path.abspath(ref_image_path)
        video_path = os.path.abspath(animate_video_path)
        output_path = os.path.splitext(video_path)[0] + "-face_swap.mp4"
        FACEFUSION_DIR = os.path.join(constants.PACKAGE_DIR, "third_party/facefusion")
        CUR_DIR = os.getcwd()
        os.chdir(FACEFUSION_DIR)
        job_dir = '.jobs'
        os.makedirs(os.path.join(job_dir, 'queued'), exist_ok=True)
        template_json = os.path.join(constants.CHECKPOINT_DIR, "facefusion_templates/omni_animate_v1.json")
        if not os.path.exists(template_json):
            hf_hub_download(repo_id=self.omni_animate_model_id,
                            subfolder="facefusion_templates",
                            filename="omni_animate_v1.json",
                            local_dir=constants.CHECKPOINT_DIR
                            )
        with open(template_json, "r") as fin:
            template_data = json.load(fin)
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
        return output_path

    @torch.no_grad()
    def __call__(
            self,
            ref_image_path,
            src_video_path,
            height: int = 576,
            width: int = 1024,
            stride: int = 1,
            num_frames: Optional[int] = None,
            tile_size: Optional[int] = 32,
            seed=1234,
            tile_overlap: Optional[int] = 6,
            num_inference_steps: int = 25,
            min_guidance_scale: float = 3.0,
            max_guidance_scale: float = 3.0,
            fps: int = 8,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.,
            image_only_indicator: bool = False,
            decode_chunk_size: Optional[int] = 8,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pt",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
            device: Union[str, torch.device] = None,
            scale_latents=False,
            use_faceswap=True,
            **kwargs
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/
                feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid`
                and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second.The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation.
                The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image,
                the higher it is the less the video will look like the init image. Increase it for more motion.
            image_only_indicator (`bool`, *optional*, defaults to False):
                Whether to treat the inputs as batch of images instead of videos.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time.The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption.
                By default, the decoder will decode all frames at once for maximal quality.
                Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            device:
                On which device the pipeline runs on.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image(
        "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        if not os.path.exists(ref_image_path) or not os.path.exists(src_video_path):
            print(ref_image_path, src_video_path)
            return None
        try:
            vcap = cv2.VideoCapture(src_video_path)
            wfps = int(vcap.get(cv2.CAP_PROP_FPS)) // stride
            vcap.release()
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            image, image_pose, org_ref_w, org_ref_h, ow, oh = self.pre_process(ref_image_path, src_video_path,
                                                                               height=height,
                                                                               width=width, stride=stride,
                                                                               **kwargs)
            num_frames = image_pose.size(0)

            num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
            decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

            generator = torch.Generator()
            generator.manual_seed(seed)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(image, height, width)

            # 2. Define call parameters
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            else:
                batch_size = image.shape[0]
            device_, dtype_ = utils.get_optimize_device()
            device = device if device is not None else device_
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            self._guidance_scale = max_guidance_scale
            # 3. Encode input image
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt,
                                                  self.do_classifier_free_guidance)

            # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
            # is why it is reduced here.
            fps = fps - 1

            # 4. Encode input image using VAE
            image = self.image_processor.preprocess(image, height=oh, width=ow).to(device)
            noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
            image = image + noise_aug_strength * noise

            image_latents = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                scale_latents=scale_latents
            )
            image_latents = image_latents.to(image_embeddings.dtype)

            # Repeat the image latents for each frame so we can concatenate them with the noise
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

            # 5. Get Added Time IDs
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(device)

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                tile_size,
                num_channels_latents,
                oh,
                ow,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
            latents = latents.repeat(1, num_frames // tile_size + 1, 1, 1, 1)[:, :num_frames]

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

            # 7. Prepare guidance scale
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
            guidance_scale = guidance_scale.to(device, latents.dtype)
            guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
            guidance_scale = _append_dims(guidance_scale, latents.ndim)

            self._guidance_scale = guidance_scale

            # 8. Denoising loop
            self._num_timesteps = len(timesteps)
            indices = [[0, *range(i + 1, min(i + tile_size, num_frames))] for i in
                       range(0, num_frames - tile_size + 1, tile_size - tile_overlap)]
            if indices:
                if indices[-1][-1] < num_frames - 1:
                    indices.append([0, *range(num_frames - tile_size + 1, num_frames)])
            else:
                indices.append([0, *range(num_frames)])
                indices[-1].extend([num_frames - 1] * (tile_size - len(indices[-1])))

            with torch.cuda.device(device):
                torch.cuda.empty_cache()

            with self.progress_bar(total=len(timesteps) * len(indices)) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Concatenate image_latents over channels dimension
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    # predict the noise residual
                    noise_pred = torch.zeros_like(image_latents)
                    noise_pred_cnt = image_latents.new_zeros((num_frames,))
                    weight = (torch.arange(tile_size, device=device) + 0.5) * 2. / tile_size
                    weight = torch.minimum(weight, 2 - weight)
                    for idx in indices:
                        # classification-free inference
                        pose_latents = self.pose_net(image_pose[idx].to(device, dtype=latents.dtype))
                        _noise_pred = self.unet(
                            latent_model_input[:1, idx],
                            t,
                            encoder_hidden_states=image_embeddings[:1],
                            added_time_ids=added_time_ids[:1],
                            pose_latents=torch.zeros_like(pose_latents),
                            image_only_indicator=image_only_indicator,
                            return_dict=False,
                        )[0]
                        noise_pred[:1, idx] += _noise_pred * weight[:, None, None, None]

                        # normal inference
                        _noise_pred = self.unet(
                            latent_model_input[1:, idx],
                            t,
                            encoder_hidden_states=image_embeddings[1:],
                            added_time_ids=added_time_ids[1:],
                            pose_latents=pose_latents,
                            image_only_indicator=image_only_indicator,
                            return_dict=False,
                        )[0]
                        noise_pred[1:, idx] += _noise_pred * weight[:, None, None, None]

                        noise_pred_cnt[idx] += weight
                        progress_bar.update()
                    noise_pred.div_(noise_pred_cnt[:, None, None, None])

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)

            if not output_type == "latent":
                frames = self.decode_latents(latents, num_frames, decode_chunk_size)
                frames = tensor2vid(frames, self.image_processor, output_type=output_type)
            else:
                frames = latents

            self.maybe_free_model_hooks()
            if not return_dict:
                return frames

            def scale_video(video, width, height):
                video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
                scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
                scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                                 width)  # [batch, frames, channels, height, width]

                return scaled_video

            if kwargs.get("keep_ref_dim", False):
                frames = scale_video(frames, org_ref_w, org_ref_h)
            frames = (frames * 255.0).to(torch.uint8)[0, 1:].permute((0, 2, 3, 1))
            date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
            result_dir = kwargs.get("result_dir", "./results/{}-{}".format(self.__class__.__name__, date_str))
            os.makedirs(result_dir, exist_ok=True)
            save_vapth = os.path.join(result_dir, os.path.basename(src_video_path))
            options = {
                'crf': '18',  # 较低的 CRF 值表示更高的质量
                'preset': 'fast',  # 较慢的预设通常会产生更好的质量
                'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
            }
            write_video(save_vapth, frames.cpu(), wfps, options=options)
            torch.cuda.empty_cache()
            # face swap
            if use_faceswap:
                save_vapth = self.post_process(ref_image_path, animate_video_path=save_vapth)
            logger.info(f"animate video path: {save_vapth}")
            return save_vapth
        except Exception as e:
            print(e.__str__())
            return None
