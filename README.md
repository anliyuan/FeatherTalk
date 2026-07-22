# FeatherTalk

**C++流式推理代码来了 ！！！！！！！！！！详情见** [**README**](https://github.com/anliyuan/FeatherTalk/blob/main/FeatherTalk-CPP/README.md)**。欢迎大家试用，star，issue。**

FeatherTalk is a lightweight audio-driven talking-head framework. It is a cleaned-up and
extended version of [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human),
focused on personalized digital-human training, smaller audio features, and mobile-friendly UNet deployment.

FeatherTalk 是一个轻量级音频驱动数字人训练与推理项目。它基于
[Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human) 整理和扩展，重点放在个人数字人训练、更小的音频特征编码器，以及更适合移动端部署的 UNet。

## Demo Preview / Demo 预览

This preview was generated with the latest FeatherHuBERT audio encoder and the matching UNet checkpoint.

这个预览使用最新的 FeatherHuBERT 音频编码器和配套 UNet checkpoint 生成。

<video src="./demo/feathertalk_demo_latest_188.mp4" controls width="100%"></video>

[Open demo video / 打开 demo 视频](./demo/feathertalk_demo_latest_188.mp4)

## Demo Training Assets / Demo 训练资源

Download the demo training asset pack from Google Drive:

[kanghui_training_video_featherhubert_188_latest.zip](https://drive.google.com/file/d/1-gSAp_BlQ7xPBDQRCf9cjaYjQT03IOxI/view?usp=drive_link)

The package contains a trainable demo video and a trained FeatherHuBERT audio encoder checkpoint.
You can use it to quickly reproduce the training flow without preparing your own video first.

After unzipping, the package contains:

```text
kanghui_training_video.MOV
feather_hubert_188_latest_99.pth
```

Preprocess the demo video with the trained FeatherHuBERT checkpoint:

```bash
cd data_utils
python process.py /path/to/kanghui_training_video.MOV \
  --asr hubert \
  --feather_hubert_checkpoint /path/to/feather_hubert_188_latest_99.pth
cd ..
```

可以从 Google Drive 下载 demo 训练资源包：

[kanghui_training_video_featherhubert_188_latest.zip](https://drive.google.com/file/d/1-gSAp_BlQ7xPBDQRCf9cjaYjQT03IOxI/view?usp=drive_link)

这个压缩包包含一个可训练的 demo 视频，以及一个已经训练好的 FeatherHuBERT 音频编码器 checkpoint。
你可以先用它快速跑通训练流程，再替换成自己的视频和模型。

解压后包含：

```text
kanghui_training_video.MOV
feather_hubert_188_latest_99.pth
```

可以直接用训练好的 FeatherHuBERT checkpoint 预处理 demo 视频：

```bash
cd data_utils
python process.py /path/to/kanghui_training_video.MOV \
  --asr hubert \
  --feather_hubert_checkpoint /path/to/feather_hubert_188_latest_99.pth
cd ..
```

## Highlights / 亮点

- Train a personalized talking-head model from a short video.
- Keep the original HuBERT and Wenet feature pipelines.
- Add **FeatherHuBERT**, a small HuBERT-compatible waveform audio encoder.
- Support both the original UNet and a MobileOne-style UNet through `--unet`.
- Add mouth ROI loss and adjacent-frame temporal mouth loss to reduce mouth jitter.
- Remove SyncNet from the default training path.
- Support checkpoint resume and ONNX export.
- Reparameterize MobileOne blocks for faster inference.

- 可以用一段较短的人脸口播视频训练个人数字人。
- 保留原来的 HuBERT 和 Wenet 音频特征流程。
- 新增 **FeatherHuBERT**：一个小型 HuBERT 兼容 waveform 音频编码器。
- 支持原始 UNet 和 MobileOne 风格 UNet，可通过 `--unet` 切换。
- 新增嘴部 ROI loss 和连续帧嘴部 temporal loss，用来缓解嘴部抖动。
- 默认训练流程不再依赖 SyncNet。
- 支持断点续训和 ONNX 导出。
- MobileOne 模型支持重参数化，方便推理加速。

## FeatherHuBERT: HuBERT-Compatible Audio Encoder / FeatherHuBERT：HuBERT 兼容音频编码器

FeatherHuBERT is one of the main changes in this version. The original HuBERT feature extractor is
accurate, but it is a large ASR model. For this talking-head pipeline, we do not need text
recognition. We only need a stable speech representation that can drive mouth motion, so
FeatherHuBERT is designed as a small feature encoder distilled from HuBERT.

FeatherHuBERT 是这一版最重要的工作之一。原始 HuBERT 的特征很好用，但它本质上是一个很大的 ASR 模型。
在这个数字人流程里，我们并不需要识别文字，只需要一组稳定、能驱动嘴部运动的语音表征。因此 FeatherHuBERT
被设计成一个从 HuBERT 蒸馏出来的小型音频特征编码器。

### Design Idea / 设计思路

- Keep the same external interface as HuBERT: 16 kHz waveform in, HuBERT-shaped feature out.
- 保持和 HuBERT 一样的外部接口：输入 16kHz waveform，输出 HuBERT 兼容特征。
- Do not use fbank. FeatherHuBERT directly consumes waveform, like HuBERT.
- 不使用 fbank。FeatherHuBERT 和 HuBERT 一样直接吃 waveform。
- Keep the same digital-human feature shape: `[audio_frames, 1024]`, reshaped to `[video_frames, 2, 1024]`.
- 保持数字人训练和推理需要的特征形状：`[audio_frames, 1024]`，再 reshape 成 `[video_frames, 2, 1024]`。
- Match HuBERT's 20 ms feature stride so the rest of the pipeline can still use `--asr hubert`.
- 对齐 HuBERT 的 20ms 特征步长，因此后面的数据集、UNet 和推理流程仍然可以使用 `--asr hubert`。
- Use a HuBERT-style strided Conv1d frontend, depthwise temporal blocks, and a 1x1 projection to 1024 channels.
- 结构上使用 HuBERT 风格的 stride Conv1d 前端、depthwise temporal blocks，以及投影到 1024 维的 1x1 卷积。

The recommended small configuration is:

推荐的小模型配置是：

```text
channels=256
num_blocks=8
output_dim=1024
parameters: ~3.36M
```

Approximate audio-side compute for 1 second of 16 kHz audio:

```text
Original HuBERT Large:     ~17.8G MACs
FeatherHuBERT 256ch/8blk:  ~0.43G MACs
```

FeatherHuBERT is roughly 40x lighter than HuBERT Large on the audio side. In the full
talking-head pipeline, the UNet is usually the mobile bottleneck, so use `--unet mobileone` and
export a reparameterized ONNX model when targeting edge devices.

在音频侧，FeatherHuBERT 相比 HuBERT Large 大约轻 40 倍。完整数字人流程中，移动端瓶颈通常会转移到
UNet，所以如果目标是端侧部署，建议使用 `--unet mobileone` 并导出重参数化后的 ONNX 模型。

### Training Recipe / 训练思路

FeatherHuBERT was trained as a HuBERT feature distillation model. The offline training recipe is:

FeatherHuBERT 是通过 HuBERT 特征蒸馏得到的。离线训练思路是：

1. Extract HuBERT teacher features from a large audio corpus.
2. Train FeatherHuBERT to regress those teacher features from waveform.

1. 先用原始 HuBERT 从大量音频中提取教师特征。
2. 再训练 FeatherHuBERT，让它从 waveform 直接回归这些教师特征。

The distillation loss combines feature MSE, cosine distance, temporal delta loss, acceleration loss,
and feature-norm loss. The temporal terms are useful because this project cares about mouth motion,
not only static feature similarity.

蒸馏 loss 由特征 MSE、cosine distance、时间一阶差分、二阶加速度差分和特征 norm 约束组成。加入时间差分项是因为
数字人更关心嘴部运动是否稳定，而不仅仅是单帧特征是否接近。

The distillation training scripts are not included in this source release. The demo training asset
pack provides a trained FeatherHuBERT checkpoint that can be used directly.

本开源版本不包含 FeatherHuBERT 的蒸馏训练脚本。上面的 demo 训练资源包已经提供了一个训练好的
FeatherHuBERT checkpoint，可以直接使用。

Check model size / 查看模型大小：

```bash
python data_utils/feather_hubert/feather_hubert.py \
  --stats \
  --channels 256 \
  --num_blocks 8 \
  --output_dim 1024
```

Use FeatherHuBERT for a training video / 用 FeatherHuBERT 替换训练视频的 HuBERT 特征：

```bash
python data_utils/feather_hubert/feather_hubert.py \
  --wav /path/to/your_video_folder/aud.wav \
  --checkpoint ./feather_hubert_ckpt/last.pth \
  --out /path/to/your_video_folder/aud_hu.npy
```

Then train or infer with `--asr hubert`, because FeatherHuBERT produces the same feature shape as
the HuBERT path.

之后训练或推理仍然使用 `--asr hubert`，因为 FeatherHuBERT 输出的是和 HuBERT 路线相同的特征形状。

## What Changed / 相比旧版优化了什么

The original `Ultralight-Digital-Human` showed that a very small personalized talking-head
model can run in real time, especially with Wenet features. FeatherTalk keeps that simple workflow
and adds a lighter HuBERT-compatible audio path plus deployment-oriented model choices.

原版 `Ultralight-Digital-Human` 已经证明轻量个人数字人可以实时运行，尤其是使用 Wenet 特征时。
FeatherTalk 保留这个简单流程，并新增更轻的 HuBERT 兼容音频路径，以及更适合部署的模型选择。

| Area / 方向 | Previous / 旧版 | FeatherTalk / 新版 |
| --- | --- | --- |
| Audio features / 音频特征 | Wenet or original HuBERT | Wenet, original HuBERT, and FeatherHuBERT |
| Audio encoder / 音频编码器 | HuBERT is accurate but heavy; Wenet is faster | FeatherHuBERT keeps HuBERT-like input/output with much lower cost |
| UNet | Original lightweight UNet | Original UNet plus MobileOne-style UNet |
| Training loss / 训练损失 | Full-frame pixel/perceptual loss | Optional mouth ROI loss and temporal mouth delta loss |
| Deployment / 部署 | Mainly Wenet path | MobileOne reparameterization and ONNX export path |
| Code hygiene / 代码整理 | Research codebase | Clean source release with a small preview demo, but without bulky data, checkpoints, or private paths |

## Recording Advice / 训练视频建议

Source video quality strongly affects the final result. Bad audio usually leads to unstable
or inaccurate mouth motion.

训练视频质量会强烈影响最终效果。声音质量差，通常会直接导致嘴型不准或嘴部抖动。

Recommended / 建议：

- 3 to 5 minutes of talking-head video is enough for a first model.
- 第一次训练可以先准备 3 到 5 分钟的人脸口播视频。
- The full face should be visible in every frame.
- 视频中每一帧都应该能看到完整人脸。
- Use clear speech without strong noise, echo, or room reverb.
- 声音要清晰，尽量避免噪声、回声和明显房间混响。
- Use an external microphone if possible.
- 条件允许的话，建议使用外接麦克风。
- Use 25 fps video for HuBERT or FeatherHuBERT features.
- HuBERT 和 FeatherHuBERT 路线建议使用 25fps 视频。
- Use 20 fps video for Wenet features.
- Wenet 路线需要使用 20fps 视频。

## Installation / 安装

Python 3.10 is recommended.

建议使用 Python 3.10。

```bash
conda create -n feathertalk python=3.10
conda activate feathertalk

pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches your CUDA version. Example:

如果要用 CUDA 训练，请安装与你 CUDA 版本匹配的 PyTorch。示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

`ffmpeg` is required for preprocessing and audio/video merging.

预处理和音视频合并需要安装 `ffmpeg`。

## Required Weights / 必需权重

The lightweight preprocessing models are included in this source release:

开源代码已经包含预处理必需的轻量模型：

```text
data_utils/scrfd_2.5g_kps.onnx          # face detector / 人脸检测
data_utils/checkpoint_epoch_335.pth.tar # landmark detector / 关键点检测
```

Wenet feature extraction still needs a large encoder file, which is not included because it is about
110 MB. Download it and place it under `data_utils/` only if you want to use `--asr wenet`:

Wenet 特征提取仍然需要一个较大的 encoder 文件，约 110MB，因此不直接放进仓库。只有使用 `--asr wenet`
时才需要下载并放到 `data_utils/` 下：

```text
data_utils/encoder.onnx                 # only needed for Wenet feature extraction
```

Download link / 下载链接：

[encoder.onnx](https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link)

Original HuBERT extraction uses Hugging Face Transformers and downloads:

原始 HuBERT 特征提取会通过 Hugging Face Transformers 下载：

```text
facebook/hubert-large-ls960-ft
```

See `docs/WEIGHTS.md` for a short checklist.

## Prepare Training Data / 准备训练数据

Put your training video in a clean folder. The preprocessing script writes outputs next to
the video.

把训练视频放进一个干净目录中。预处理脚本会把结果写在视频同级目录下。

```text
your_video_folder/
  train.mp4
  aud.wav
  aud_hu.npy              # HuBERT or FeatherHuBERT mode
  full_body_img/
  landmarks/
```

HuBERT mode / HuBERT 模式：

```bash
cd data_utils
python process.py /path/to/your_video_folder/train.mp4 --asr hubert
cd ..
```

FeatherHuBERT mode / FeatherHuBERT 模式：

```bash
cd data_utils
python process.py /path/to/your_video_folder/train.mp4 \
  --asr hubert \
  --feather_hubert_checkpoint /path/to/feather_hubert.pth
cd ..
```

Wenet mode / Wenet 模式：

```bash
cd data_utils
python process.py /path/to/your_video_folder/train.mp4 --asr wenet
cd ..
```

HuBERT and FeatherHuBERT use the `hubert` feature shape in this project, so they use the
25fps path. Wenet uses 20fps.

HuBERT 和 FeatherHuBERT 在本项目中使用同一种 `hubert` 特征形状，因此走 25fps 路线。Wenet 走 20fps 路线。

## Train / 训练

Original UNet / 原始 UNet：

```bash
python train.py \
  --dataset_dir /path/to/your_video_folder \
  --save_dir ./checkpoint \
  --asr hubert \
  --unet original \
  --epochs 200 \
  --batchsize 16
```

MobileOne UNet / MobileOne 版 UNet：

```bash
python train.py \
  --dataset_dir /path/to/your_video_folder \
  --save_dir ./checkpoint_mobileone \
  --asr hubert \
  --unet mobileone \
  --epochs 200 \
  --batchsize 16
```

Mouth ROI loss / 嘴部 ROI loss：

```bash
python train_mouth_roi_loss.py \
  --dataset_dir /path/to/your_video_folder \
  --save_dir ./checkpoint_mouth_roi \
  --asr hubert \
  --unet original \
  --epochs 200 \
  --mouth_weight 4.0
```

Mouth ROI + temporal loss / 嘴部 ROI + 连续帧 temporal loss：

```bash
python train_mouth_roi_temporal_loss.py \
  --dataset_dir /path/to/your_video_folder \
  --save_dir ./checkpoint_mouth_temporal \
  --asr hubert \
  --unet original \
  --epochs 200 \
  --batchsize 8 \
  --mouth_weight 4.0 \
  --temporal_weight 0.5 \
  --temporal_mouth_weight 4.0
```

Resume training / 断点续训：

```bash
python train.py \
  --dataset_dir /path/to/your_video_folder \
  --save_dir ./checkpoint \
  --asr hubert \
  --unet original \
  --epochs 300 \
  --resume ./checkpoint/last.pth
```

## Inference / 推理

First extract audio features, then run video inference.

先提取测试音频特征，再运行视频推理。

Original HuBERT / 原始 HuBERT：

```bash
python data_utils/hubert.py \
  --wav /path/to/test.wav \
  --out /path/to/test_hu.npy
```

FeatherHuBERT：

```bash
python data_utils/feather_hubert/feather_hubert.py \
  --wav /path/to/test.wav \
  --checkpoint ./feather_hubert_ckpt/last.pth \
  --out /path/to/test_hu.npy
```

Run inference / 运行推理：

```bash
python inference.py \
  --asr hubert \
  --unet original \
  --dataset /path/to/your_video_folder \
  --audio_feat /path/to/test_hu.npy \
  --checkpoint ./checkpoint/199.pth \
  --save_path ./result.mp4 \
  --audio_wav /path/to/test.wav
```

MobileOne checkpoint / MobileOne checkpoint：

```bash
python inference.py \
  --asr hubert \
  --unet mobileone \
  --dataset /path/to/your_video_folder \
  --audio_feat /path/to/test_hu.npy \
  --checkpoint ./checkpoint_mobileone/199.pth \
  --save_path ./result_mobileone.mp4 \
  --audio_wav /path/to/test.wav
```

MobileOne checkpoints are reparameterized for inference by default.

MobileOne checkpoint 在推理时默认会做重参数化。

## Export ONNX / 导出 ONNX

Original UNet / 原始 UNet：

```bash
python pth2onnx.py \
  --checkpoint ./checkpoint/199.pth \
  --onnx_path ./feathertalk_unet.onnx \
  --asr hubert \
  --unet original
```

MobileOne UNet / MobileOne 版 UNet：

```bash
python pth2onnx.py \
  --checkpoint ./checkpoint_mobileone/199.pth \
  --onnx_path ./feathertalk_mobileone.onnx \
  --asr hubert \
  --unet mobileone
```

MobileOne export uses the reparameterized inference graph by default.

MobileOne 导出时默认使用重参数化后的推理图。

## C++ Offline Inference / C++ 离线推理

`FeatherTalk-CPP/` provides a standalone macOS/Apple Silicon C++ runner for
the latest FeatherHuBERT + FeatherTalk UNet path. It runs the 16 kHz waveform
encoder and UNet directly through MNN with CPU, Metal, or OpenCL backends. MNN
models are converted with FP16 convolution weights, and output frames are piped
to `ffmpeg` without creating a large temporary image sequence.

`FeatherTalk-CPP/` 提供了可直接运行的 macOS/Apple Silicon C++ 离线推理版本。它通过
MNN 的 CPU、Metal 或 OpenCL 后端直接运行 16kHz waveform 的 FeatherHuBERT 和 UNet。MNN
模型使用 FP16 卷积权重，输出帧会直接管道传给 `ffmpeg`，不会产生大量临时图片。

See [FeatherTalk-CPP/README.md](./FeatherTalk-CPP/README.md) for dependency setup,
checkpoint export, and the complete run command.

完整依赖安装、checkpoint 导出和运行命令见
[FeatherTalk-CPP/README.md](./FeatherTalk-CPP/README.md)。

## Streaming Inference / 流式推理

`dihuman_run.py` now supports both the legacy Wenet streaming encoder and the new
FeatherHuBERT-compatible lightweight path.

`dihuman_run.py` 现在同时支持旧版 Wenet 流式 encoder，以及新的 FeatherHuBERT 轻量模型路径。

Wenet streaming / Wenet 流式：

```bash
python dihuman_run.py \
  --asr wenet \
  --data_path /path/to/your_video_folder \
  --unet_onnx ./wenet_unet.onnx \
  --encoder_onnx ./encoder.onnx \
  --audio_wav /path/to/test.wav \
  --out_video ./stream_video.mp4 \
  --out_audio ./stream_audio.wav \
  --video_size 1280 720
```

FeatherHuBERT + HuBERT-shape UNet / FeatherHuBERT + HuBERT 形状 UNet：

```bash
python dihuman_run.py \
  --asr feather_hubert \
  --data_path /path/to/your_video_folder \
  --unet_onnx ./feathertalk_mobileone.onnx \
  --feather_hubert_checkpoint ./feather_hubert_ckpt/last.pth \
  --feather_right_context_frames 4 \
  --audio_wav /path/to/test.wav \
  --out_video ./stream_video.mp4 \
  --out_audio ./stream_audio.wav \
  --video_size 1280 720
```

The FeatherHuBERT path consumes 16 kHz waveform directly and feeds the UNet with
HuBERT-compatible audio tensors shaped `[1, 16, 32, 32]`. It uses 25 fps by default:
two 20 ms HuBERT tokens are grouped into one video frame.

FeatherHuBERT 路线直接输入 16kHz waveform，并向 UNet 输入 HuBERT 兼容的 `[1, 16, 32, 32]`
音频张量。默认视频帧率是 25fps：两个 20ms HuBERT token 对应一帧视频。

`--feather_right_context_frames` controls the small look-ahead used before emitting
FeatherHuBERT features. The default `4` is a stability-first setting; set it to `0` for lower
latency when you are profiling real-time behavior.

`--feather_right_context_frames` 控制 FeatherHuBERT 特征输出前等待的右上下文帧数。默认 `4`
更偏稳定；如果要压低实时延迟，可以设成 `0` 做性能测试。

For a quick Python demo, pass a `.pth` checkpoint through `--feather_hubert_checkpoint`.
For deployment, export FeatherHuBERT to ONNX separately and pass it with `--encoder_onnx`;
the expected encoder ONNX output is `[1, audio_tokens, 1024]`.

快速 Python demo 可以直接通过 `--feather_hubert_checkpoint` 加载 `.pth`。部署时可以把
FeatherHuBERT 单独导出为 ONNX，然后通过 `--encoder_onnx` 传入；encoder ONNX 的期望输出是
`[1, audio_tokens, 1024]`。

Merge audio and video / 合并音视频：

```bash
ffmpeg -i stream_video.mp4 -i stream_audio.wav -c:v libx264 -c:a aac result_stream.mp4
```

## File Overview / 文件结构

```text
data_utils/process.py                              preprocessing / 数据预处理
data_utils/hubert.py                               original HuBERT feature extraction / 原始 HuBERT 特征
data_utils/wenet_infer.py                          Wenet feature extraction / Wenet 特征
data_utils/feather_hubert/                         FeatherHuBERT feature extraction / FeatherHuBERT 特征提取
train.py                                           baseline training / 基础训练
train_mouth_roi_loss.py                            mouth ROI loss training / 嘴部 ROI loss 训练
train_mouth_roi_temporal_loss.py                   mouth ROI + temporal loss training / 连续帧 loss 训练
inference.py                                       offline video inference / 离线推理
pth2onnx.py                                        ONNX export / ONNX 导出
unet.py                                            original lightweight UNet / 原始轻量 UNet
unet_mobileone.py                                  MobileOne-style UNet / MobileOne 版 UNet
model_factory.py                                   original/mobileone model selector / 模型选择器
dihuman_run.py                                     Wenet + FeatherHuBERT streaming inference / Wenet + FeatherHuBERT 流式推理
FeatherTalk-CPP/                                    standalone C++ offline inference / 独立 C++ 离线推理
```

## Notes / 说明

- This is still a research-oriented project. Good video and clean audio matter more than
  almost any single training trick.
- 这个项目仍然偏研究和实验性质。好的训练视频和干净音频，比单个训练 trick 更重要。
- FeatherHuBERT is distributed here as an inference-time audio feature encoder. Its distillation
  training code is not part of this source release.
- FeatherHuBERT 在本项目中作为推理/训练数据预处理阶段的音频特征编码器使用；它的蒸馏训练代码不包含在本开源版本中。
- For mobile deployment, start with MobileOne UNet + reparameterized ONNX + FP16, then try
  INT8 if quality is acceptable.
- 移动端部署建议从 MobileOne UNet + 重参数化 ONNX + FP16 开始，如果效果可接受再尝试 INT8。
- Large datasets, checkpoints, generated experiment outputs, and pretrained binary weights are
  intentionally excluded from this source release. The small demo video under `demo/` is included
  only as a preview.
- 本开源代码有意不包含大型数据集、checkpoint、实验生成结果和大型预训练权重。`demo/` 下的小 demo 视频仅用于效果预览。
## 歪比巴卜！！！！点个star吧！！！！

## License / 许可证

Apache License 2.0. See `LICENSE`.
