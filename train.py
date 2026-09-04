"""Train FeatherTalk with mouth-focused and optional temporal losses."""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mouth_roi import MouthRoiConfig, TemporalMouthRoiDataset
from model import IMAGE_SIZE, Model, load_checkpoint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FeatherTalk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init_from",
        type=str,
        default="",
        help="optional checkpoint used for shape-compatible visual-weight initialization",
    )
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--see_res", action="store_true")
    parser.add_argument("--see_res_dir", type=str, default="./train_tmp_img_mouth_temporal")

    parser.add_argument("--mouth_weight", type=float, default=4.0)
    parser.add_argument("--temporal_weight", type=float, default=0.5)
    parser.add_argument("--temporal_mouth_weight", type=float, default=4.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)
    parser.add_argument(
        "--mouth_gradient_weight",
        type=float,
        default=0.5,
        help="mouth-only first-order high-frequency reconstruction loss",
    )
    parser.add_argument(
        "--temporal_lowpass_kernel",
        type=int,
        default=5,
        help="odd blur kernel applied before temporal losses; 1 disables low-pass",
    )
    parser.add_argument("--temporal_stride", type=int, default=1)
    parser.add_argument("--mouth_start", type=int, default=90)
    parser.add_argument("--mouth_end", type=int, default=110)
    parser.add_argument("--mouth_expand_x", type=float, default=1.45)
    parser.add_argument("--mouth_expand_y", type=float, default=1.75)
    parser.add_argument("--mouth_min_w", type=int, default=52)
    parser.add_argument("--mouth_min_h", type=int, default=36)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    temporal_lowpass_kernel = getattr(args, "temporal_lowpass_kernel", 1)
    if temporal_lowpass_kernel < 1 or temporal_lowpass_kernel % 2 == 0:
        raise ValueError("--temporal_lowpass_kernel must be a positive odd integer")


def cuda_is_healthy() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        torch.cuda.set_device(0)
        probe = torch.ones((16, 16), device="cuda")
        _ = probe @ probe
        torch.cuda.synchronize()
        return True, torch.cuda.get_device_name(0)
    except Exception as exc:
        return False, repr(exc)


def get_training_device(allow_cpu: bool) -> torch.device:
    cuda_ok, cuda_info = cuda_is_healthy()
    if cuda_ok:
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        f"CUDA is not healthy: {cuda_info}. Pass --allow_cpu for a CPU test."
    )


class PerceptualLoss:
    CONV_3_3_LAYER = 14

    def __init__(self, criterion: nn.Module, device: torch.device):
        self.criterion = criterion
        cnn = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT).features
        self.content = nn.Sequential(*list(cnn.children())[: self.CONV_3_3_LAYER + 1])
        self.content = self.content.to(device).eval()
        for parameter in self.content.parameters():
            parameter.requires_grad_(False)

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        generated_features = self.content(generated)
        with torch.no_grad():
            target_features = self.content(target)
        return self.criterion(generated_features, target_features)


def save_checkpoint(
    path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int
) -> None:
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )


def resume_if_any(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> int:
    if not path:
        return 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0


def mouth_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype)
    numerator = ((prediction - target).abs() * mask).sum()
    denominator = mask.sum().clamp_min(1.0) * prediction.shape[1]
    return numerator / denominator


def temporal_delta_losses(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
    pixel_criterion: nn.Module,
    lowpass_kernel: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lowpass_kernel > 1:
        batch_size, pair_len = preds.shape[:2]
        flat_preds = preds.reshape(batch_size * pair_len, *preds.shape[2:])
        flat_labels = labels.reshape(batch_size * pair_len, *labels.shape[2:])
        padding = lowpass_kernel // 2
        preds_for_temporal = F.avg_pool2d(
            flat_preds, lowpass_kernel, stride=1, padding=padding
        ).reshape_as(preds)
        labels_for_temporal = F.avg_pool2d(
            flat_labels, lowpass_kernel, stride=1, padding=padding
        ).reshape_as(labels)
    else:
        preds_for_temporal = preds
        labels_for_temporal = labels
    pred_delta = preds_for_temporal[:, 1] - preds_for_temporal[:, 0]
    label_delta = labels_for_temporal[:, 1] - labels_for_temporal[:, 0]
    union_mask = mouth_masks.max(dim=1).values
    loss_temporal = pixel_criterion(pred_delta, label_delta)
    loss_temporal_mouth = mouth_l1_loss(pred_delta, label_delta, union_mask)
    return loss_temporal, loss_temporal_mouth


def mouth_gradient_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
) -> torch.Tensor:
    """Match horizontal/vertical detail inside the mouth ROI only."""
    masks = mouth_masks.to(dtype=preds.dtype)
    pred_dx = preds[..., 1:] - preds[..., :-1]
    label_dx = labels[..., 1:] - labels[..., :-1]
    mask_dx = masks[..., 1:] * masks[..., :-1]
    pred_dy = preds[..., 1:, :] - preds[..., :-1, :]
    label_dy = labels[..., 1:, :] - labels[..., :-1, :]
    mask_dy = masks[..., 1:, :] * masks[..., :-1, :]
    numerator = ((pred_dx - label_dx).abs() * mask_dx).sum()
    numerator = numerator + ((pred_dy - label_dy).abs() * mask_dy).sum()
    denominator = (mask_dx.sum() + mask_dy.sum()).clamp_min(1.0) * preds.shape[1]
    return numerator / denominator


def compute_total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
    pixel_criterion: nn.Module,
    perceptual_loss: PerceptualLoss,
    mouth_weight: float,
    temporal_weight: float,
    temporal_mouth_weight: float,
    perceptual_weight: float,
    mouth_gradient_weight: float = 0.0,
    temporal_lowpass_kernel: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, pair_len = preds.shape[:2]
    flat_preds = preds.reshape(batch_size * pair_len, *preds.shape[2:])
    flat_labels = labels.reshape(batch_size * pair_len, *labels.shape[2:])
    flat_masks = mouth_masks.reshape(batch_size * pair_len, *mouth_masks.shape[2:])

    loss_pixel = pixel_criterion(flat_preds, flat_labels)
    loss_mouth = mouth_l1_loss(flat_preds, flat_labels, flat_masks)
    if temporal_weight != 0.0 or temporal_mouth_weight != 0.0:
        loss_temporal, loss_temporal_mouth = temporal_delta_losses(
            preds,
            labels,
            mouth_masks,
            pixel_criterion,
            lowpass_kernel=temporal_lowpass_kernel,
        )
    else:
        # Independent clips may contain hard cuts.  When temporal supervision is
        # disabled, avoid both the invalid cross-cut constraint and its compute.
        loss_temporal = flat_preds.new_zeros(())
        loss_temporal_mouth = flat_preds.new_zeros(())
    loss_perceptual = perceptual_loss(flat_preds, flat_labels)
    loss_mouth_gradient = mouth_gradient_loss(flat_preds, flat_labels, flat_masks)
    total = (
        loss_pixel
        + mouth_weight * loss_mouth
        + temporal_weight * loss_temporal
        + temporal_mouth_weight * loss_temporal_mouth
        + perceptual_weight * loss_perceptual
        + mouth_gradient_weight * loss_mouth_gradient
    )
    return total, {
        "full": float(loss_pixel.detach().cpu()),
        "mouth": float(loss_mouth.detach().cpu()),
        "temp": float(loss_temporal.detach().cpu()),
        "temp_mouth": float(loss_temporal_mouth.detach().cpu()),
        "percep": float(loss_perceptual.detach().cpu()),
        "mouth_grad": float(loss_mouth_gradient.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    pixel_criterion: nn.Module,
    perceptual_loss: PerceptualLoss,
    device: torch.device,
    progress_desc: str,
    dataset_len: int,
    mouth_weight: float,
    temporal_weight: float,
    temporal_mouth_weight: float,
    perceptual_weight: float,
    mouth_gradient_weight: float = 0.0,
    temporal_lowpass_kernel: int = 1,
):
    net.train()
    with tqdm(total=dataset_len * 2, desc=progress_desc, unit="frame") as progress:
        for batch in loader:
            imgs, labels, audio_feat, mouth_masks = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            audio_feat = audio_feat.to(device)
            mouth_masks = mouth_masks.to(device)
            batch_size, pair_len = imgs.shape[:2]
            flat_imgs = imgs.reshape(batch_size * pair_len, *imgs.shape[2:])
            flat_audio = audio_feat.reshape(batch_size * pair_len, *audio_feat.shape[2:])

            flat_preds = net(flat_imgs, flat_audio)
            preds = flat_preds.reshape(batch_size, pair_len, *flat_preds.shape[1:])

            loss, parts = compute_total_loss(
                preds,
                labels,
                mouth_masks,
                pixel_criterion,
                perceptual_loss,
                mouth_weight,
                temporal_weight,
                temporal_mouth_weight,
                perceptual_weight,
                mouth_gradient_weight,
                temporal_lowpass_kernel,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            progress.set_postfix(
                {
                    "loss": parts["total"],
                    "full": parts["full"],
                    "mouth": parts["mouth"],
                    "temp": parts["temp"],
                    "temp_m": parts["temp_mouth"],
                    "grad": parts["mouth_grad"],
                }
            )
            progress.update(batch_size * pair_len)


def _to_img(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def dump_sample(net: nn.Module, dataset: TemporalMouthRoiDataset, save_dir: str, epoch: int, device: torch.device):
    net.eval()
    idx = random.randint(0, len(dataset) - 1)
    imgs, labels, audio_feat, mouth_masks = dataset[idx]
    with torch.no_grad():
        preds = net(imgs.to(device), audio_feat.to(device))
    panels = [
        _to_img(preds[0]),
        _to_img(labels[0]),
        _to_img(preds[1]),
        _to_img(labels[1]),
    ]
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_pred0_real0_pred1_real1.jpg"), np.concatenate(panels, axis=1))
    mask_img = (mouth_masks.max(dim=0).values.numpy()[0] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_mouth_union_mask.jpg"), mask_img)


def main():
    args = parse_args()
    validate_args(args)
    device = get_training_device(args.allow_cpu)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.see_res:
        os.makedirs(args.see_res_dir, exist_ok=True)

    mouth_config = MouthRoiConfig(
        start=args.mouth_start,
        end=args.mouth_end,
        expand_x=args.mouth_expand_x,
        expand_y=args.mouth_expand_y,
        min_w=args.mouth_min_w,
        min_h=args.mouth_min_h,
    )
    dataset = TemporalMouthRoiDataset(
        args.dataset_dir,
        mouth_config=mouth_config,
        temporal_stride=args.temporal_stride,
        image_size=IMAGE_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    net = Model().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pixel_criterion = nn.L1Loss()
    perceptual_loss = PerceptualLoss(nn.MSELoss(), device=device)

    if args.init_from and args.resume:
        raise ValueError("Use either --init_from or --resume, not both.")
    if args.init_from:
        source = load_checkpoint_state(args.init_from, device)
        current = net.state_dict()
        matched = {
            key: value
            for key, value in source.items()
            if key in current and current[key].shape == value.shape
        }
        current.update(matched)
        net.load_state_dict(current)
        print(
            f"[init] loaded {len(matched)}/{len(current)} shape-compatible tensors "
            f"from {args.init_from}"
        )

    start_epoch = resume_if_any(args.resume, net, optimizer, device)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            net,
            loader,
            optimizer,
            pixel_criterion,
            perceptual_loss,
            device,
            progress_desc=f"Epoch {epoch + 1}/{args.epochs}",
            dataset_len=len(dataset),
            mouth_weight=args.mouth_weight,
            temporal_weight=args.temporal_weight,
            temporal_mouth_weight=args.temporal_mouth_weight,
            perceptual_weight=args.perceptual_weight,
            mouth_gradient_weight=args.mouth_gradient_weight,
            temporal_lowpass_kernel=args.temporal_lowpass_kernel,
        )

        is_save_epoch = (epoch + 1) % args.save_every == 0
        if is_save_epoch or epoch == args.epochs - 1:
            save_checkpoint(os.path.join(args.save_dir, f"{epoch}.pth"), net, optimizer, epoch)
            save_checkpoint(os.path.join(args.save_dir, "last.pth"), net, optimizer, epoch)

        if args.see_res:
            dump_sample(net, dataset, args.see_res_dir, epoch, device)


if __name__ == "__main__":
    main()
