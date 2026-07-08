"""Train original UNet with an additional mouth ROI reconstruction loss."""

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

from dataset_mouth_roi import MouthRoiConfig, MouthRoiDataset
from model_factory import UNET_CHOICES, create_model
from train import PerceptualLoss, get_training_device, resume_if_any, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train original UNet with mouth ROI weighted loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--asr", type=str, default="hubert", choices=["wenet", "hubert"])
    parser.add_argument("--unet", type=str, default="original", choices=UNET_CHOICES)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--see_res", action="store_true")
    parser.add_argument("--see_res_dir", type=str, default="./train_tmp_img_mouth_roi")

    parser.add_argument("--mouth_weight", type=float, default=4.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)
    parser.add_argument("--mouth_start", type=int, default=90)
    parser.add_argument("--mouth_end", type=int, default=110)
    parser.add_argument("--mouth_expand_x", type=float, default=1.45)
    parser.add_argument("--mouth_expand_y", type=float, default=1.75)
    parser.add_argument("--mouth_min_w", type=int, default=52)
    parser.add_argument("--mouth_min_h", type=int, default=36)
    return parser.parse_args()


def mouth_l1_loss(preds: torch.Tensor, labels: torch.Tensor, mouth_masks: torch.Tensor) -> torch.Tensor:
    masks = mouth_masks.to(dtype=preds.dtype)
    diff = (preds - labels).abs() * masks
    denom = masks.sum().clamp_min(1.0) * preds.shape[1]
    return diff.sum() / denom


def compute_total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mouth_masks: torch.Tensor,
    pixel_criterion: nn.Module,
    perceptual_loss: PerceptualLoss,
    mouth_weight: float,
    perceptual_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_pixel = pixel_criterion(preds, labels)
    loss_mouth = mouth_l1_loss(preds, labels, mouth_masks)
    loss_perceptual = perceptual_loss(preds, labels)
    total = loss_pixel + mouth_weight * loss_mouth + perceptual_weight * loss_perceptual
    return total, {
        "full": float(loss_pixel.detach().cpu()),
        "mouth": float(loss_mouth.detach().cpu()),
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
    perceptual_weight: float,
):
    net.train()
    with tqdm(total=dataset_len, desc=progress_desc, unit="img") as progress:
        for imgs, labels, audio_feat, mouth_masks in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            audio_feat = audio_feat.to(device)
            mouth_masks = mouth_masks.to(device)

            preds = net(imgs, audio_feat)
            loss, parts = compute_total_loss(
                preds,
                labels,
                mouth_masks,
                pixel_criterion,
                perceptual_loss,
                mouth_weight,
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
                }
            )
            progress.update(imgs.shape[0])


def dump_sample(net: nn.Module, dataset: MouthRoiDataset, save_dir: str, epoch: int, device: torch.device):
    net.eval()
    idx = random.randint(0, len(dataset) - 1)
    img_concat_T, target_T, audio_feat, mouth_mask_T = dataset[idx]
    img_concat_T = img_concat_T[None].to(device)
    audio_feat = audio_feat[None].to(device)
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
    pred_img = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    real_img = (target_T.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    mask_img = (mouth_mask_T.numpy()[0] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}.jpg"), pred_img)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_real.jpg"), real_img)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_mouth_mask.jpg"), mask_img)


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
    dataset = MouthRoiDataset(args.dataset_dir, args.asr, mouth_config=mouth_config)
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
