#!/usr/bin/env python3
"""Export the released FeatherTalk inference checkpoints to ONNX.

The generated models use the exact interfaces consumed by the C++ runner:
  FeatherHuBERT: [1, samples] -> [1, tokens, 1024]
  UNet:          [1, 6, 160, 160], [1, 16, 32, 32] -> [1, 3, 160, 160]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_utils.feather_hubert.feather_hubert import load_feather_hubert
from model_factory import (
    UNET_CHOICES,
    create_model,
    load_checkpoint_state,
    maybe_reparameterize_mobileone,
)


def export_feather(checkpoint: Path, output: Path, opset: int) -> None:
    model = load_feather_hubert(checkpoint, device=torch.device("cpu")).eval()
    waveform = torch.zeros((1, 16000), dtype=torch.float32)
    torch.onnx.export(
        model,
        waveform,
        output,
        input_names=["waveform"],
        output_names=["hidden"],
        dynamic_axes={"waveform": {1: "samples"}, "hidden": {1: "tokens"}},
        opset_version=opset,
        export_params=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    print(f"[export] FeatherHuBERT -> {output}")


def export_unet(checkpoint: Path, output: Path, opset: int, unet: str = "original") -> None:
    model = create_model("hubert", unet).eval()
    model.load_state_dict(load_checkpoint_state(str(checkpoint), torch.device("cpu")))
    # MobileOne 训练期为多分支结构（rbr_conv/rbr_scale/rbr_skip），导出前重参数化为等价单分支卷积
    model = maybe_reparameterize_mobileone(model, unet).eval()
    image = torch.zeros((1, 6, 160, 160), dtype=torch.float32)
    audio = torch.zeros((1, 16, 32, 32), dtype=torch.float32)
    torch.onnx.export(
        model,
        (image, audio),
        output,
        input_names=["input", "audio"],
        output_names=["output"],
        opset_version=opset,
        export_params=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    print(f"[export] FeatherTalk UNet -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feather-checkpoint", type=Path, required=True)
    parser.add_argument("--unet-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "FeatherTalk-CPP" / "models")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--unet", type=str, default="original", choices=UNET_CHOICES,
                        help="UNet 变体；mobileone 会在导出前重参数化为单分支卷积")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feather_output = args.output_dir / "feather_hubert.onnx"
    unet_output = args.output_dir / "unet_hubert.onnx"
    export_feather(args.feather_checkpoint, feather_output, args.opset)
    export_unet(args.unet_checkpoint, unet_output, args.opset, args.unet)


if __name__ == "__main__":
    main()
