c++流式推理在这了，如果大家有将这个项目部署到移动端的需求，需要自己去搞各端的代码，接进去就可以了。按照我自己的测试，当前这个项目跑在中端及以上的移动端设备上肯定是没问题了。音频编码器已经快的飞起了，实测本地部署要比wenet快10倍以上，后续我的优化重点会放在unet上了。目前还有一些提速的小方法，大概还能将unet提速个百分之二三十，我还在测，后续也会开源到这个项目。欢迎大家试用，star，issue。以下是codex帮我写的⬇️。

# FeatherTalk C++ / FeatherTalk C++ 离线部署

This directory contains the standalone macOS / Apple Silicon deployment path
for FeatherTalk. It runs the released FeatherHuBERT and UNet models with MNN,
without Python, PyTorch, OpenCV, or ONNX Runtime during inference.

这个目录提供 FeatherTalk 的 macOS / Apple Silicon 独立部署路径。推理阶段使用
MNN 运行 FeatherHuBERT 和 UNet，不依赖 Python、PyTorch、OpenCV 或 ONNX Runtime。


```text
16 kHz waveform -> FeatherHuBERT MNN -> [audio_tokens, 1024]
                                      -> [video_frames, 2, 1024]
image + 8-frame audio window -> FeatherTalk UNet MNN -> output frame
```

`ffmpeg` receives rendered BGR frames through a pipe and writes the final MP4
with the source WAV. No temporary image sequence is written to disk.

程序将渲染后的 BGR 图像直接通过管道交给 `ffmpeg`，再与输入 WAV 合成为 MP4；不会在
磁盘上生成大量临时图片帧。

## Highlights / 特性

- Native ARM64 MNN runtime with CPU, Metal, and OpenCL backends.
- FeatherHuBERT and UNet are converted with FP16 convolution weights.
- The CPU runner automatically uses MNN's ARM CPU extension when available.
- Keeps the Python inference audio contract: normalized 16 kHz waveform in,
  `[1, T, 1024]` features out.
- Supports the same 8-frame audio window, face crop, mouth mask, and frame
  bounce order as FeatherTalk Python inference.

- 原生 ARM64 MNN 运行时，支持 CPU、Metal 和 OpenCL 后端。
- FeatherHuBERT 和 UNet 在转换时使用 FP16 卷积权重。
- CPU 推理会在可用时自动使用 MNN 的 ARM CPU Extension。
- 保持与 Python 推理相同的音频接口：归一化 16kHz waveform 输入，输出
  `[1, T, 1024]` 特征。
- 保持与 FeatherTalk Python 推理相同的 8 帧音频窗口、人脸裁剪、嘴部遮罩和帧往返顺序。

## Requirements / 环境要求

This deployment path currently targets macOS on Apple Silicon. Command Line
Tools and `ffmpeg` must be available. `setup_mnn_macos.sh` installs `cmake`
and `ninja` through `python3 -m pip` only when they are not already on `PATH`.

当前部署路径面向 Apple Silicon 的 macOS。需要安装 Command Line Tools 和 `ffmpeg`。
如果系统 `PATH` 中没有 `cmake` 与 `ninja`，`setup_mnn_macos.sh` 会通过
`python3 -m pip` 安装它们。

```bash
xcode-select --install
brew install ffmpeg
```

## Build MNN and the C++ Binaries / 编译 MNN 和 C++ 程序

Run the following commands from this directory. The first script downloads the
image decoder header, the second script downloads and builds MNN with CPU,
Metal, OpenCL, and `MNNConvert`, and the last script builds both binaries.

在本目录执行下面的命令。第一个脚本下载图片解码头文件，第二个脚本下载并编译带有 CPU、
Metal、OpenCL 和 `MNNConvert` 的 MNN，最后一个脚本编译推理和测速程序。

```bash
cd /Users/anqi/Desktop/FeatherTalk/FeatherTalk-CPP
chmod +x setup_macos.sh setup_mnn_macos.sh convert_mnn_models.sh build_macos.sh

./setup_macos.sh
./setup_mnn_macos.sh
./build_macos.sh
```

The resulting binaries are:

编译完成后会得到：

```text
bin/feathertalk_mnn    # video inference / 视频推理
bin/benchmark_mnn      # model benchmark / 模型测速
```

## Export and Convert Models / 导出并转换模型

MNN conversion uses ONNX only as an offline interchange format. The C++
inference binary itself has no ONNX Runtime dependency.

MNN 转换阶段会使用 ONNX 作为离线中间格式；C++ 推理程序本身不依赖 ONNX Runtime。

Run the checkpoint export once from the FeatherTalk repository root. The paths
below use the latest local 188 checkpoints used by this project.

在 FeatherTalk 仓库根目录执行一次 checkpoint 导出。下面的路径使用本项目当前的
188 checkpoint。

```bash
/opt/anaconda3/envs/dihuman/bin/python FeatherTalk-CPP/tools/export_models.py \
  --feather-checkpoint demo_outputs/latest_188/feather_hubert_188_latest_99.pth \
  --unet-checkpoint demo_outputs/latest_188/unet_188_latest_last.pth
```

Then convert the two exported models to FP16 MNN models:

然后将两个导出的模型转换为 FP16 MNN 模型：

```bash
cd /Users/anqi/Desktop/FeatherTalk/FeatherTalk-CPP
./convert_mnn_models.sh
```

The generated files are ignored by Git:

生成的文件已被 Git 忽略：

```text
models/feather_hubert.onnx
models/unet_hubert.onnx
models/feather_hubert.mnn
models/unet_hubert.mnn
```

`convert_mnn_models.sh` passes `--fp16`, so MNN stores convolution weights in
FP16. The current release produces approximately a 6.5 MB FeatherHuBERT model
and a 23 MB original-UNet model.

`convert_mnn_models.sh` 会传入 `--fp16`，因此 MNN 以 FP16 保存卷积权重。当前版本
转换后，FeatherHuBERT 约为 6.5 MB，原始 UNet 约为 23 MB。

## Run Inference / 运行推理

The dataset directory must contain FeatherTalk preprocessing output:
`full_body_img/`, `landmarks/`, and a 16 kHz mono WAV. The command below uses
Metal with FP16 compute, the recommended Apple Silicon configuration.

数据目录需要包含 FeatherTalk 预处理结果：`full_body_img/`、`landmarks/`，以及一段
16kHz 单声道 WAV。下面的命令使用 Metal 和 FP16 计算，是 Apple Silicon 上推荐的配置。

```bash
cd /Users/anqi/Desktop/FeatherTalk/FeatherTalk-CPP

./bin/feathertalk_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/unet_hubert.mnn \
  --dataset /path/to/your_video_folder \
  --audio /path/to/your_video_folder/aud.wav \
  --output /tmp/feathertalk_mnn_demo.mp4 \
  --backend metal \
  --precision low
```

For a 60-frame smoke test, append `--max-frames 60`.

如需只跑 60 帧验证流程，在命令末尾加上 `--max-frames 60`。

### Keep Individual Frames / 保留逐帧图片

Add `--frames-dir` to write every rendered frame as a six-digit PNG while the
same frames are still streamed to `ffmpeg` for MP4 creation.

添加 `--frames-dir` 后，程序会在继续通过 `ffmpeg` 生成 MP4 的同时，把每一帧保存为六位
编号的 PNG 图片。

```bash
./bin/feathertalk_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/unet_hubert.mnn \
  --dataset /path/to/your_video_folder \
  --audio /path/to/your_video_folder/aud.wav \
  --output /tmp/feathertalk_mnn_demo.mp4 \
  --frames-dir /tmp/feathertalk_frames \
  --backend metal \
  --precision low
```

The directory contains `000000.png`, `000001.png`, and subsequent frames.

目录中会生成 `000000.png`、`000001.png` 等连续帧。

### CPU Inference / CPU 推理

Use the following command for a single-thread CPU run. MNN selects the ARM CPU
extension automatically when the hardware supports it.

如需单线程 CPU 推理，可使用下面的命令。硬件支持时，MNN 会自动选择 ARM CPU Extension。

```bash
./bin/feathertalk_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/unet_hubert.mnn \
  --dataset /path/to/your_video_folder \
  --audio /path/to/your_video_folder/aud.wav \
  --output /tmp/feathertalk_cpu_demo.mp4 \
  --backend cpu \
  --precision low \
  --threads 1
```

`--backend` accepts `cpu`, `metal`, or `opencl`. The program rejects a silent
GPU-to-CPU fallback. `--threads` controls the CPU thread count; it is accepted
by MNN GPU configurations but does not map to GPU work in the same way.

`--backend` 可选 `cpu`、`metal` 或 `opencl`。程序会拒绝 GPU 静默回退到 CPU。
`--threads` 控制 CPU 线程数；GPU 后端也会接收该参数，但它并不等价于 GPU 工作线程数。

## Benchmark / 性能测速

`benchmark_mnn` measures model forward time only. It excludes image I/O, face
crop / paste, and video encoding. The FeatherHuBERT input is `[1, 720]`, which
produces two 20 ms tokens and corresponds to one 25 fps video-frame feature.

`benchmark_mnn` 只测模型前向耗时，不包含读图、人脸裁剪 / 粘贴和视频编码。FeatherHuBERT
输入为 `[1, 720]`，会生成两个 20ms token，对应一个 25fps 视频帧的音频特征。

```bash
./bin/benchmark_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/unet_hubert.mnn \
  --backend cpu \
  --precision low \
  --threads 1 \
  --warmup 20 \
  --iterations 100
```

### FeatherHuBERT vs. Wenet / FeatherHuBERT 与 Wenet 对比

The following numbers were measured on this Apple Silicon Mac with MNN FP16,
one CPU thread, 20 warmup runs, and 100 timed runs. Wenet's fbank is required
preprocessing and runs on CPU.

下表是在当前 Apple Silicon Mac 上测得：MNN FP16、单 CPU 线程、热身 20 次、正式
测量 100 次。Wenet 的 fbank 是必需的 CPU 预处理步骤。

| Item / 项目 | Mean time / 平均耗时 |
| --- | ---: |
| FeatherHuBERT `[1,720] -> [1,2,1024]` | `0.250 ms` |
| Wenet encoder `[1,1,67,80] -> [1,16,512]` | `2.697 ms` |
| Wenet fbank `[11040] -> [67,80]` | `0.291 ms` |
| Wenet audio frontend total / Wenet 音频前端总计 | `2.988 ms` |

FeatherHuBERT is about **11.95x faster** than the complete Wenet audio
frontend in this single-call benchmark.

在这个单次音频编码调用的基准中，FeatherHuBERT 比完整 Wenet 音频前端约快 **11.95 倍**。

The input windows and output frame counts of the two encoders differ, so this
is an audio-frontend invocation comparison rather than a direct linear
conversion to the same audio duration.

两个编码器的输入窗口和一次输出的特征帧数不同，因此这是音频前端单次调用对比，不能直接
线性换算成完全相同音频时长的对比。

To benchmark an MNN-converted original Wenet encoder, add
`--wenet-model /path/to/wenet_encoder.mnn`. The comparison benchmark fixes
Wenet's stream offset only to make its graph static; this does not change its
operator count. Its fbank cost is still measured separately.

如需对比转换后的原始 Wenet MNN 编码器，可以加上
`--wenet-model /path/to/wenet_encoder.mnn`。对比程序仅固定 Wenet 的 stream offset
以便将图静态化，不改变算子数量；fbank 耗时仍需单独计算。

## Compatibility / 接口兼容性

- Input audio is normalized 16 kHz PCM waveform, matching
  `data_utils/feather_hubert/feather_hubert.py`.
- FeatherHuBERT output is `[1, T, 1024]`. Odd tokens are dropped, then each two
  tokens form one 25 fps video-frame feature.
- Each UNet call receives Python's same audio window `[i-4, ..., i+3]`, reshaped
  to `[1, 16, 32, 32]`.
- Image preprocessing follows `face_utils.py`: a 168px face crop, 160px inner
  region, and the same mouth mask geometry.

- 输入音频为归一化后的 16kHz PCM waveform，与
  `data_utils/feather_hubert/feather_hubert.py` 保持一致。
- FeatherHuBERT 输出为 `[1, T, 1024]`。会先丢弃末尾奇数 token，再每两个 token 合成一个
  25fps 视频帧的特征。
- 每次 UNet 调用接收与 Python 相同的音频窗口 `[i-4, ..., i+3]`，reshape 后为
  `[1, 16, 32, 32]`。
- 图片预处理遵循 `face_utils.py`：168px 人脸裁剪、160px 内部区域，以及相同的嘴部遮罩位置。

The released checkpoint is the original UNet. A reparameterized MobileOne UNet
can use the same C++ runner after conversion to the same two-input MNN
interface: `[1, 6, 160, 160]` image input and `[1, 16, 32, 32]` audio input.

当前发布的 checkpoint 是原始 UNet。重参数化后的 MobileOne UNet 只要转换为相同的双输入
MNN 接口，也可以复用这个 C++ 推理程序：图像输入 `[1, 6, 160, 160]`，音频输入
`[1, 16, 32, 32]`。
