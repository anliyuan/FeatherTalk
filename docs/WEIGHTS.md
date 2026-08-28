# Model weights

The preprocessing path uses these small local models:

```text
data_utils/scrfd_2.5g_kps.onnx
data_utils/checkpoint_epoch_335.pth.tar
```

The generic audio encoder is included at the repository root:

```text
feather_hubert.pth       generic audio encoder, 3.36M parameters
```

Its SHA-256 checksum is:

```text
3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46
```

Training produces a personalized 144x144 visual checkpoint for each identity.
For C++ inference, convert that checkpoint and the included audio encoder with
`FeatherTalk-CPP/tools/export_models.py` and `convert_mnn_models.sh`.

Do not commit personalized checkpoints, generated ONNX files or MNN files.
Publish approved deployment artifacts as versioned release assets.
