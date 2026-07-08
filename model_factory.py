from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


UNET_CHOICES = ("original", "mobileone")


def create_model(mode: str, unet: str = "original") -> nn.Module:
    if unet == "original":
        from unet import Model
    elif unet == "mobileone":
        from unet_mobileone import Model
    else:
        raise ValueError(f"Unknown UNet variant: {unet}. Expected one of {UNET_CHOICES}.")
    return Model(6, mode)


def load_checkpoint_state(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def maybe_reparameterize_mobileone(model: nn.Module, unet: str, enabled: bool = True) -> nn.Module:
    if unet != "mobileone" or not enabled:
        return model
    from unet_mobileone import reparameterize_model

    return reparameterize_model(model)
