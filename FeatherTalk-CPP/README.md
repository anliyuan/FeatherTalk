# FeatherTalk C++ / MNN

This directory contains the standalone C++ inference path for the released
FeatherTalk model. Runtime inference depends on MNN and FFmpeg, but not Python,
PyTorch, OpenCV or ONNX Runtime.

## Fixed model interface

```text
FeatherHuBERT
  waveform: [1, samples]
  hidden:   [1, tokens, 1024]

FeatherTalk visual model
  input:  [1, 6, 144, 144]
  audio:  [1, 40, 1024]
  output: [1, 3, 144, 144]
```

The runner extracts audio features once for the complete WAV. Each video frame
uses the corresponding 40-token slice. The generated patch is composited with a
fixed mouth soft mask.

## Export and convert models

From the repository root:

```bash
python FeatherTalk-CPP/tools/export_models.py \
  --feather-checkpoint /path/to/feather_hubert.pth \
  --visual-checkpoint /path/to/visual_model.pth

cd FeatherTalk-CPP
./setup_mnn_macos.sh
./convert_mnn_models.sh
```

This produces:

```text
models/feather_hubert.mnn
models/feathertalk_144_fp16.mnn
```

ONNX is used only as an offline conversion format.

## Build on Apple Silicon

```bash
brew install ffmpeg
./setup_macos.sh
./setup_mnn_macos.sh
./build_macos.sh
```

Generated binaries:

```text
bin/feathertalk_mnn
bin/benchmark_mnn
```

For integration into another C++ project, link the `feathertalk_runtime` CMake
target and include `include/feathertalk_api.h`:

```cpp
feathertalk::OfflineOptions options;
options.feather_model = "/models/feather_hubert.mnn";
options.visual_model = "/models/feathertalk_144_fp16.mnn";
options.dataset = "/resources/person";
options.audio_wav = "/audio/input.wav";
options.output_mp4 = "/output/result.mp4";

std::string error;
if (!feathertalk::RunOffline(options, &error)) {
  // Handle error.
}
```

## Run

The resource directory must contain matching numeric image and landmark files:

```text
person_resource/
  full_body_img/0.jpg
  full_body_img/1.jpg
  landmarks/0.lms
  landmarks/1.lms
```

```bash
./bin/feathertalk_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/feathertalk_144_fp16.mnn \
  --dataset /path/to/person_resource \
  --audio /path/to/audio_16k_mono.wav \
  --output result.mp4 \
  --backend cpu \
  --precision low \
  --threads 1
```

Useful optional flags:

```text
--max-frames N
--frames-dir PATH
--video-crf N
--profile
```

## Benchmark

```bash
./bin/benchmark_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/feathertalk_144_fp16.mnn \
  --backend cpu --precision low --threads 1
```

Generated models, MNN source/build trees and binaries are intentionally ignored
by Git.
