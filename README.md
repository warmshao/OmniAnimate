## OminiAnimate: 标题？
OmniAnimate是一个集成了多个Controllable Character Animate模型的python库，包括MimicMotion、MusePose以及自研的模型。扩充？


### 功能特性
* 支持多个SOTA的Character Animate模型，包括MimicMotion、MusePose等，持续集成中。
* 支持 pip install 直接安装使用。
* pose 预处理已经集成到piepline中，无需额外处理。
* 基于facefusion的faceswap后处理已经集成到piepline中，无需额外处理。
* 同时支持webui、api等使用方式

### 环境安装
* 推荐使用miniforge安装pyhon环境，`conda create -n omnianiamte python=3.10`, 然后
* 根据自己的系统和CUDA版本安装对应的pytorch版本
* `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
* `conda install conda-forge::onnxruntime`
* `pip install -r requirements.txt`