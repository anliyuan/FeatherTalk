# FeatherTalk

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg">
  <a href="https://github.com/anliyuan/FeatherTalk/stargazers"><img src="https://img.shields.io/github/stars/anliyuan/FeatherTalk?style=social"></a>
</p>

## 更新：本次更新重新设计了模型结构。加入了新的audio adapter，大幅提升效果，同时保证低耗时运行。我也精简了代码。有问题或优化计划或部署计划的朋友可以扫码加微信群交流。enjoy 🎉

FeatherTalk is a lightweight personalized talking-head project. Give it a short
25 FPS video of one person, train a dedicated model, and then drive that person
with any speech audio.

FeatherTalk 是一个轻量级个人数字人项目。准备一段单人口播视频，训练一个专属模型，
之后就可以用任意语音驱动这个人物说话。支持部署在移动端

## Demo / 效果

两侧使用相同的人物资源和同一段 60 秒音频。左侧是之前的旧版本，
右侧是当前更新后版本。视频对下半脸做了同比例放大，方便观察唇形和融合效果。视觉上少了很多抖动，很多音节嘴形也更准了。

<video src="./demo/feathertalk_comparison_144.mp4" controls width="100%"></video>

[打开对比视频 / Open comparison video](./demo/feathertalk_comparison_144.mp4)

## Why FeatherTalk? / 为什么选它

- **效果更好**：同时利用前后语音上下文，并针对嘴部细节和连续帧稳定性训练。
- **模型足够小**：当前视觉模型输入为 144×144，参数量约 **5.46M**。并且我自己重新训练了一个全新的超轻量级音频编码器，参数量计算量都非常小。
- **推理更高效**：FeatherHuBERT 对整段音频只计算一次，视频推理直接复用音频特征。当然，根据需要，也可以改为流式推理。
- **真正可部署**：同时提供 Python 训练/推理和 C++/MNN 接入路线，可纯 CPU 运行。可轻松部署移动端。

## Quick Start / 快速开始

### 1. 安装

Python 3.10 和 FFmpeg 是必需的。训练推荐使用 CUDA GPU。

```bash
git clone https://github.com/anliyuan/FeatherTalk.git
cd FeatherTalk

conda create -n feathertalk python=3.10
conda activate feathertalk
pip install -r requirements.txt
```

仓库根目录已经包含 `feather_hubert.pth` 音频编码器权重，无需额外下载。
其他权重说明见 [docs/WEIGHTS.md](docs/WEIGHTS.md)。

### 2. 准备训练视频

推荐使用 3–5 分钟、25 FPS 的单人口播视频。人脸需要完整可见，音频尽量清晰、无明显噪声和回声。

```bash
python data_utils/process.py /path/to/person/train.mp4 \
  --feather_hubert_checkpoint ./feather_hubert.pth
```

预处理会在视频所在目录生成训练需要的图像、关键点和音频特征。

### 3. 训练

```bash
python train.py \
  --dataset_dir /path/to/person \
  --save_dir /path/to/checkpoints \
  --epochs 200 \
  --batchsize 16
```

### 4. 提取测试音频特征

```bash
python data_utils/feather_hubert/feather_hubert.py \
  --wav /path/to/test.wav \
  --checkpoint ./feather_hubert.pth \
  --out /path/to/test_hu.npy
```

### 5. 生成视频

```bash
python inference.py \
  --dataset /path/to/person \
  --audio_feat /path/to/test_hu.npy \
  --audio_wav /path/to/test.wav \
  --checkpoint /path/to/checkpoints/199.pth \
  --save_path result.mp4
```

## C++ / MNN

C++ 版本可直接读取 WAV，在 MNN 中完成音频编码、视觉模型推理和回贴合成，
运行时不依赖 Python、PyTorch、OpenCV 或 ONNX Runtime。

模型转换、编译和工程接入说明见
[FeatherTalk-CPP/README.md](FeatherTalk-CPP/README.md)。

## Notes / 注意

- 这是个性化模型：每个人物需要单独训练一份视觉权重。
- 训练视频和收音质量会直接影响最终效果。
- 仓库内置通用 `feather_hubert.pth`；个性化 checkpoint、ONNX 和 MNN 模型不直接提交。

## License

Apache-2.0. If FeatherTalk helps you, a star is always appreciated. 🎉

## Community / 交流群

欢迎扫码加入 UDH 数字人交流群，一起交流训练、推理、C++ 部署和二次开发。
群二维码有效期有限，过期后会在这里更新。

<p align="center">
  <img src="./assets/wechat_group_qr.jpg" alt="UDH 数字人交流群二维码" width="360">
</p>
