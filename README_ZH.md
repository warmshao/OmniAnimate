# OmniAnimate: 统一的可控角色动画视频生成框架
<a href="README_ZH.md">中文</a> | <a href="README.md">English</a>

OmniAnimate 是一个统一的可控角色动画视频生成框架，集成了多个前沿的可控角色动画生成模型，包括 MimicMotion、MusePose 以及自研模型。
本框架提供了一站式的可控角色动画视频生成解决方案，专注于可控角色动画视频生成模型的部署。

<video src="https://github.com/user-attachments/assets/247fec20-54b6-4496-bef4-92dff16f5180" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

> **注意**：demo视频中的部分功能仅在专业版中提供。

## 🌟 核心特性

- 集成多个前沿角色动画模型，包括 [MimicMotion](https://github.com/Tencent/MimicMotion)、[MusePose](https://github.com/TMElyralab/MusePose) 等， 更多模型正在集成中。
- 内置强大的姿态预处理模型，确保最佳生成效果
- 集成基于 [FaceFusion](https://github.com/facefusion/facefusion) 的换脸后处理功能
- 多样化使用方式：Python API、网页界面、REST API
- 支持 pip 一键安装，自动下载所需模型

## 🚀 安装说明

### 环境要求

推荐使用 miniforge 管理 Python 环境：

```bash
conda create -n omnianiamte python=3.10
conda activate omnianiamte
```

### 安装步骤

1. 根据您的系统和 CUDA 版本从 [PyTorch 官网](https://pytorch.org) 安装对应的 PyTorch>2.0 版本
   根据您的系统和 CUDA 版本安装对应的 PyTorch：

   **Windows/Linux 用户：**
   - CUDA 11.8:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - CUDA 12.1:
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - CPU 版本（不推荐）:
     ```bash
     pip3 install torch torchvision torchaudio
     ```

   **MacOS 用户：**
   ```bash
   pip3 install torch torchvision torchaudio
   ```
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 安装 OmniAnimate：
   ```bash
   pip install git+https://github.com/warmshao/OmniAnimate.git
   ```

## 💻 使用方法

### Python API 调用

```python
from omni_animate.pipelines.pipeline_mimic_motion import MimicMotionPipeline

# 设置输入路径
ref_image_path = "assets/examples/img.png"
src_video_path = "assets/examples/001.mp4"

# 初始化处理流程
pipe = MimicMotionPipeline()

# 生成动画视频
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

### 网页界面

启动网页服务：

```bash
python webui.py --host 127.0.0.1 --port 6006
```

<video src="https://github.com/user-attachments/assets/d7fa1838-7d42-4033-854c-271c47cf3c02" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>


## 🌟 专业版特性

专业版在开源版本的基础上提供了显著的性能提升和更多功能：

### 性能优化
- 采用 TensorRT 加速，处理速度提升 3 倍以上
- 使用更大规模数据集训练的优化模型
- 更高质量的动画生成效果

<video src="https://github.com/user-attachments/assets/c07ce9af-6ff4-4f87-ae61-dc90eb29fa12" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

<video src="https://github.com/user-attachments/assets/bc5dc5ab-ca62-4572-97f0-e484768294fa" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>


### 扩展功能
- 动物舞蹈视频生成
- 视频风格迁移
- 视频人物替换
- 更多特色功能持续开发中

## 📧 联系我们
专业版欢迎加我的discord(warmshao)或微信咨询。

<img src="assets/wx/alex.jpg" alt="微信" width="300" height="340">
