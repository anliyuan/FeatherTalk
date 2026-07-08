# FeatherTalk Pretrained Weights

This source release intentionally excludes binary model weights.

Place the required files here before running preprocessing or Wenet inference:

```text
data_utils/scrfd_2.5g_kps.onnx
data_utils/checkpoint_epoch_335.pth.tar
data_utils/encoder.onnx
```

HuBERT extraction uses Hugging Face Transformers and the model id:

```text
facebook/hubert-large-ls960-ft
```

FeatherHuBERT checkpoints are regular PyTorch `.pth` files. They are not included in the source
tree and are ignored by git by default. You can use the checkpoint from the demo training asset
pack or place your own externally trained checkpoint wherever you prefer.
