"""Train original UNet with mouth ROI and adjacent-frame delta losses."""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mouth_roi import MouthRoiConfig, TemporalMouthRoiDataset
from model_factory import UNET_CHOICES, create_model
from train import PerceptualLoss, get_training_device, resume_if_any, save_checkpoint
from train_mouth_roi_loss import mouth_l1_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original UNet with mouth ROI and temporal delta losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--asr", type=str, default="hubert", choices=["wenet", "hubert"])
    parser.add_argument("--unet", type=str, default="original", choices=UNET_CHOICES)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--see_res", action="store_true")
    parser.add_argument("--see_res_dir", type=str, default="./train_tmp_img_mouth_temporal")

    parser.add_argument("--mouth_weight", type=float, default=4.0)
    parser.add_argument("--temporal_weight", type=float, default=0.5)
    parser.add_argument("--temporal_mouth_weight", type=float, default=4.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)
    parser.add_argument("--temporal_stride", type=int, default=1)

    parser.add_argument("--mouth_start", type=int, default=90)
    parser.add_argument("--mouth_end", type=int, default=110)
    parser.add_argument("--mouth_expand_x", type=float, default=1.45)
    parser.add_argument("--mouth_expand_y", type=float, default=1.75)
    parser.add_argument("--mouth_min_w", type=int, default=52)
    parser.add_argument("--mouth_min_h", type=int, default=36)
    return parser.parse_args()


def temporal_delta_losses(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
    pixel_criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_delta = preds[:, 1] - preds[:, 0]
    label_delta = labels[:, 1] - labels[:, 0]
    union_mask = mouth_masks.max(dim=1).values
    loss_temporal = pixel_criterion(pred_delta, label_delta)
    loss_temporal_mouth = mouth_l1_loss(pred_delta, label_delta, union_mask)
    return loss_temporal, loss_temporal_mouth


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
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, pair_len = preds.shape[:2]
    flat_preds = preds.reshape(batch_size * pair_len, *preds.shape[2:])
    flat_labels = labels.reshape(batch_size * pair_len, *labels.shape[2:])
    flat_masks = mouth_masks.reshape(batch_size * pair_len, *mouth_masks.shape[2:])

    loss_pixel = pixel_criterion(flat_preds, flat_labels)
    loss_mouth = mouth_l1_loss(flat_preds, flat_labels, flat_masks)
    loss_temporal, loss_temporal_mouth = temporal_delta_losses(
        preds,
        labels,
        mouth_masks,
        pixel_criterion,
    )
    loss_perceptual = perceptual_loss(flat_preds, flat_labels)
    total = (
        loss_pixel
        + mouth_weight * loss_mouth
        + temporal_weight * loss_temporal
        + temporal_mouth_weight * loss_temporal_mouth
        + perceptual_weight * loss_perceptual
    )
    return total, {
        "full": float(loss_pixel.detach().cpu()),
        "mouth": float(loss_mouth.detach().cpu()),
        "temp": float(loss_temporal.detach().cpu()),
        "temp_mouth": float(loss_temporal_mouth.detach().cpu()),
        "percep": float(loss_perceptual.detach().cpu()),
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
):
    net.train()
    with tqdm(total=dataset_len * 2, desc=progress_desc, unit="frame") as progress:
        for imgs, labels, audio_feat, mouth_masks in loader:
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
        args.asr,
        mouth_config=mouth_config,
        temporal_stride=args.temporal_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    net = create_model(args.asr, args.unet).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pixel_criterion = nn.L1Loss()
    perceptual_loss = PerceptualLoss(nn.MSELoss(), device=device)

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
        )

        is_save_epoch = (epoch + 1) % args.save_every == 0
        if is_save_epoch or epoch == args.epochs - 1:
            save_checkpoint(os.path.join(args.save_dir, f"{epoch}.pth"), net, optimizer, epoch)
            save_checkpoint(os.path.join(args.save_dir, "last.pth"), net, optimizer, epoch)

        if args.see_res:
            dump_sample(net, dataset, args.see_res_dir, epoch, device)


if __name__ == "__main__":
    main()
