# FeatherTalk

FeatherTalk is a compact, personalized audio-driven talking-head system. The
public codebase contains one supported pipeline so that training, Python
inference and C++ deployment all use the same model contract.

FeatherTalk 是一个轻量级个人数字人项目。开源版只保留一条正式路线，
训练、Python 推理和 C++ 部署使用完全一致的输入输出协议。

## Model contract

```text
16 kHz WAV
  -> FeatherHuBERT: [1, samples] -> [1, tokens, 1024]
  -> per-frame audio window: [1, 40, 1024]

reference crop + masked current crop: [1, 6, 144, 144]
  -> FeatherTalk visual model
  -> generated crop: [1, 3, 144, 144]
  -> mouth soft blend into the resource frame
```

The audio encoder processes the full WAV once. For each output frame, inference
only slices the already-computed feature sequence. The 40 tokens correspond to
20 video frames at 25 FPS.

## Installation

Python 3.10 is recommended. FFmpeg is required for preprocessing and audio/video
muxing.

```bash
conda create -n feathertalk python=3.10
conda activate feathertalk
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build matching the machine's CUDA
version.

## Prepare a training resource

Use a 25 FPS talking-head video and a FeatherHuBERT checkpoint:

```bash
cd data_utils
python process.py /path/to/train.mp4 \
  --feather_hubert_checkpoint /path/to/feather_hubert.pth
cd ..
```

The resource directory will contain:

```text
person_resource/
  aud.wav
  aud_hu.npy
  full_body_img/
  landmarks/
```

## Train

```bash
python train.py \
  --dataset_dir /path/to/person_resource \
  --save_dir /path/to/checkpoints \
  --epochs 200 \
  --batchsize 16
```

The training objective combines full-crop reconstruction, mouth ROI,
adjacent-frame temporal, perceptual and optional mouth-gradient losses. See
`python train.py --help` for their weights.

## Python inference

```bash
python inference.py \
  --dataset /path/to/person_resource \
  --audio_feat /path/to/aud_hu.npy \
  --audio_wav /path/to/aud.wav \
  --checkpoint /path/to/visual_model.pth \
  --save_path result.mp4
```

## C++ / MNN deployment

The C++ runner takes a WAV directly, runs FeatherHuBERT and the visual model with
MNN, blends each generated mouth patch and streams frames to FFmpeg.

See [FeatherTalk-CPP/README.md](FeatherTalk-CPP/README.md) for model conversion,
building and runtime commands.

## Required weights

The repository contains only the small face detector and landmark detector used
by preprocessing. FeatherHuBERT and personalized visual checkpoints should be
distributed as versioned release assets rather than committed to Git.

## License

Apache-2.0. Third-party weights and runtimes retain their own licenses; verify
their redistribution terms before publishing a release package.
