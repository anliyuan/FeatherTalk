"""共用的人脸预处理 / 音频窗口工具。

数据加载、训练、推理、流式推理里都需要做的几件事：
- 从 .lms 文件读取关键点
- 根据关键点算嘴部下方的正方形 bbox
- 把 bbox 区域裁出来、resize 到固定尺寸、再取里层喂给网络
- 在音频特征序列上以当前帧为中心取一个上下文窗口

这些动作之前在多个文件里复制粘贴，集中在这里以保证一致性。
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np
import torch


# 固定的 144x144 模型协议：先 resize 到 152x152，再取中心 144x144。
FACE_CROP_SIZE = 152
FACE_INNER_SIZE = 144
FACE_BORDER = 4
GEOMETRY_BASE_SIZE = 160

# 涂黑嘴部矩形 (x, y, w, h)
MASK_RECT = (5, 5, 150, 145)

# 取 [i-10, i+9] 共20个视频帧，每帧对应两个音频 token。
AUDIO_HALF_WINDOW = 10


Bbox = Tuple[int, int, int, int]


def read_landmarks(lms_path: str) -> np.ndarray:
    """从 .lms 文件读关键点。每行 'x y'，返回 int32 ndarray [N, 2]。"""
    pts = []
    with open(lms_path, "r") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            pts.append(np.fromstring(line, sep=" ", dtype=np.float32))
    return np.array(pts, dtype=np.int32)


def compute_face_bbox(landmarks: np.ndarray) -> Bbox:
    """从关键点算出包含嘴部的正方形 bbox (xmin, ymin, xmax, ymax)。

    选点规则沿用原作者：横向以 #1 / #31 关键点为左右边界，纵向以 #52 为上边界，
    并强制为正方形（边长等于横向宽度）。
    """
    xmin = int(landmarks[1][0])
    ymin = int(landmarks[52][1])
    xmax = int(landmarks[31][0])
    width = xmax - xmin
    ymax = ymin + width
    return xmin, ymin, xmax, ymax


def crop_face(
    img: np.ndarray, bbox: Bbox, crop_size: int = FACE_CROP_SIZE
) -> np.ndarray:
    """裁切 bbox 区域并 resize 到指定的正方形尺寸。"""
    xmin, ymin, xmax, ymax = bbox
    region = img[ymin:ymax, xmin:xmax]
    return cv2.resize(region, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def extract_inner(
    face_crop: np.ndarray, inner_size: int = FACE_INNER_SIZE
) -> np.ndarray:
    """从正方形 crop 中取中心的指定尺寸区域。"""
    border = (face_crop.shape[0] - inner_size) // 2
    return face_crop[border:border + inner_size, border:border + inner_size].copy()


def mask_mouth(img: np.ndarray, inner_size: int = FACE_INNER_SIZE) -> np.ndarray:
    """把指定尺寸的人脸图嘴部区域按比例涂黑（原地修改并返回）。"""
    scale = inner_size / float(GEOMETRY_BASE_SIZE)
    x, y, width, height = MASK_RECT
    rect = (
        round(x * scale),
        round(y * scale),
        round(width * scale),
        round(height * scale),
    )
    return cv2.rectangle(img, rect, (0, 0, 0), -1)


def mouth_soft_blend_mask(image_size: int = FACE_INNER_SIZE) -> torch.Tensor:
    """Return a tighter feathered mask around the moving mouth region.

    The mask keeps generated pixels around the moving mouth and preserves the
    current resource frame around the cheeks, chin and crop boundary.
    """
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    nx = (xx + 0.5) / float(image_size)
    ny = (yy + 0.5) / float(image_size)
    dx = (nx - 0.5) / 0.46
    dy = (ny - 0.37) / 0.34
    radius = np.sqrt(dx * dx + dy * dy)
    edge_t = np.clip((radius - 0.82) / (1.0 - 0.82), 0.0, 1.0)
    alpha = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t)
    return torch.from_numpy(alpha[None].astype(np.float32))


def mouth_fill_mask(image_size: int = FACE_INNER_SIZE) -> torch.Tensor:
    """Return the hard mouth input rectangle used for base-frame filling."""
    scale = image_size / float(GEOMETRY_BASE_SIZE)
    x, y, width, height = MASK_RECT
    x1 = round(x * scale)
    y1 = round(y * scale)
    x2 = min(image_size, x1 + round(width * scale))
    y2 = min(image_size, y1 + round(height * scale))
    mask = np.zeros((1, image_size, image_size), dtype=np.float32)
    mask[:, y1:y2, x1:x2] = 1.0
    return torch.from_numpy(mask)


def hwc_to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """[H, W, 3] uint8 → [3, H, W] float32 tensor，并归一化到 [0, 1]。"""
    img = img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(img)


def gather_audio_window(
    features: np.ndarray,
    index: int,
    half_window: int = AUDIO_HALF_WINDOW,
) -> torch.Tensor:
    """取以 index 为中心、半径 half_window 的音频特征窗口，越界处用 0 填充。

    返回 shape [2*half_window, ...] 的 tensor，dtype 跟 features 保持一致。
    """
    left = index - half_window
    right = index + half_window
    pad_left = max(0, -left)
    pad_right = max(0, right - features.shape[0])
    left = max(0, left)
    right = min(features.shape[0], right)
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        padding = torch.zeros(
            (pad_left, *features.shape[1:]), dtype=auds.dtype
        )
        auds = torch.cat([padding, auds], dim=0)
    if pad_right > 0:
        padding = torch.zeros(
            (pad_right, *features.shape[1:]), dtype=auds.dtype
        )
        auds = torch.cat([auds, padding], dim=0)
    return auds


def reshape_audio_feat(
    audio_feat: torch.Tensor,
    mode: str = "hubert",
    raw_hubert: bool = True,
) -> torch.Tensor:
    """Convert 20 video-frame features into ``[40, 1024]`` tokens."""
    if mode != "hubert" or not raw_hubert:
        raise ValueError("FeatherTalk requires HuBERT-compatible raw features")
    if audio_feat.ndim != 3 or audio_feat.shape[-2:] != (2, 1024):
        raise ValueError("audio features must have shape [frames, 2, 1024]")
    return audio_feat.reshape(-1, 1024).contiguous()


def count_jpgs(dir_path: str) -> int:
    return sum(1 for f in os.listdir(dir_path) if f.endswith(".jpg"))


def load_face_crop(
    img_path: str, lms_path: str, crop_size: int = FACE_CROP_SIZE
) -> np.ndarray:
    """便捷函数：读图 + 读关键点 + 算 bbox + 裁切，返回 FACE_CROP_SIZE 大小的 crop。"""
    img = cv2.imread(img_path)
    bbox = compute_face_bbox(read_landmarks(lms_path))
    return crop_face(img, bbox, crop_size=crop_size)
