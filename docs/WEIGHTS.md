# FeatherTalk Pretrained Weights

This source release includes the lightweight preprocessing models required for face detection and
landmark detection:

```text
data_utils/scrfd_2.5g_kps.onnx
data_utils/checkpoint_epoch_335.pth.tar
```

Wenet feature extraction still needs the larger encoder file below. It is about 110 MB, so it is
not tracked in git. Download it only if you want to use `--asr wenet`:

```text
data_utils/encoder.onnx
```

Download link:

[encoder.onnx](https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link)

HuBERT extraction uses Hugging Face Transformers and the model id:

```text
facebook/hubert-large-ls960-ft
```

FeatherHuBERT checkpoints are regular PyTorch `.pth` files. They are not included in the source
tree and are ignored by git by default. You can use the checkpoint from the demo training asset
pack or place your own externally trained checkpoint wherever you prefer.
