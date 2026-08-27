# Model weights

The preprocessing path uses these small local models:

```text
data_utils/scrfd_2.5g_kps.onnx
data_utils/checkpoint_epoch_335.pth.tar
```

Training and inference additionally require:

```text
feather_hubert.pth       generic audio encoder
visual_model.pth         personalized 144x144 visual checkpoint
```

For C++ inference, convert the two checkpoints with
`FeatherTalk-CPP/tools/export_models.py` and `convert_mnn_models.sh`.

Do not commit generated checkpoints, ONNX files or MNN files. Publish approved
weights as versioned release assets with checksums and third-party notices.
