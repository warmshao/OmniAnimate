# OmniAnimate: Unified Controllable Character Animation Framework
<a href="README_ZH.md">中文</a> | <a href="README.md">English</a>

OmniAnimate is a unified framework for controllable character animation video generation, integrating multiple cutting-edge controllable character animation models, including MimicMotion, MusePose, and proprietary models.
This framework provides a one-stop solution for controllable character animation video generation, focusing on the deployment of controllable character animation video generation models.

<video src="https://github.com/user-attachments/assets/247fec20-54b6-4496-bef4-92dff16f5180" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

> **Note**: Some features shown in the demo video are only available in the premium version.

## 🌟 Key Features

- Integration of multiple cutting-edge character animation models, including [MimicMotion](https://github.com/Tencent/MimicMotion), [MusePose](https://github.com/TMElyralab/MusePose), and more models being integrated.
- Built-in powerful pose preprocessing model ensuring optimal generation results
- Integrated face-swapping post-processing based on [FaceFusion](https://github.com/facefusion/facefusion)
- Multiple usage options: Python API, Web UI, REST API
- One-click installation via pip with automatic model downloading

## 🚀 Installation

### Prerequisites

We recommend using miniforge for Python environment management:

```bash
conda create -n omnianiamte python=3.10
conda activate omnianiamte
```

### Installation Steps

1. Install PyTorch>2.0 according to your system and CUDA version from [PyTorch official website](https://pytorch.org):

   **Windows/Linux Users:**
   - CUDA 11.8:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - CUDA 12.1:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - CPU version (not recommended):
     ```bash
     pip3 install torch torchvision torchaudio
     ```

   **MacOS Users:**
   ```bash
   pip3 install torch torchvision torchaudio
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install OmniAnimate:
   ```bash
   pip install git+https://github.com/warmshao/OmniAnimate.git
   ```

## 💻 Usage

### Python API

```python
from omni_animate.pipelines.pipeline_mimic_motion import MimicMotionPipeline

# Set input paths
ref_image_path = "assets/examples/img.png"
src_video_path = "assets/examples/001.mp4"

# Initialize pipeline
pipe = MimicMotionPipeline()

# Generate animated video
animated_video = pipe(
    ref_image_path, 
    src_video_path,
    tile_size=32,            
    tile_overlap=6,          
    height=1024,             
    width=576,               
    stride=1,                
    fps=7,                   
    noise_aug_strength=0,    
    num_inference_steps=25,  
    seed=1234,              
    min_guidance_scale=2,    
    max_guidance_scale=2,    
    decode_chunk_size=8,     
    use_faceswap=True      
)
```

### WebUI

Launch the web service:

```bash
python webui.py --host 127.0.0.1 --port 6006
```

<video src="https://github.com/user-attachments/assets/d7fa1838-7d42-4033-854c-271c47cf3c02" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

## 🌟 Premium Version Features

The premium version offers significant performance improvements and additional features beyond the open-source version:

### Performance Optimizations
- TensorRT acceleration providing 3x+ speedup
- Optimized models trained on larger datasets
- Higher quality animation generation

<video src="https://github.com/user-attachments/assets/c07ce9af-6ff4-4f87-ae61-dc90eb29fa12" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

<video src="https://github.com/user-attachments/assets/bc5dc5ab-ca62-4572-97f0-e484768294fa" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

### Extended Features
- Animal dance video generation
- Video style transfer
- Video character replacement
- More features under development

## 📧 Contact Us
For premium version inquiries, feel free to contact me on Discord (warmshao) or WeChat.

<img src="assets/wx/alex.jpg" alt="WeChat" width="300" height="340">
