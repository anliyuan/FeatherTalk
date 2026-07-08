"""Dataset variant that also returns a mouth ROI mask for weighted losses."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from face_utils import (
    FACE_BORDER,
    FACE_CROP_SIZE,
    FACE_INNER_SIZE,
    compute_face_bbox,
    count_jpgs,
    extract_inner,
    gather_audio_window,
    hwc_to_chw_tensor,
    load_face_crop,
    mask_mouth,
    read_landmarks,
    reshape_audio_feat,
)


_AUDIO_FEAT_FILE = {
    "wenet": "aud_wenet.npy",
    "hubert": "aud_hu.npy",
}


@dataclass(frozen=True)
class MouthRoiConfig:
    start: int = 90
    end: int = 110
    expand_x: float = 1.45
    expand_y: float = 1.75
    min_w: int = 52
    min_h: int = 36
    pad: int = 2


def _project_mouth_points_to_inner(landmarks: np.ndarray, config: MouthRoiConfig) -> np.ndarray:
    bbox = compute_face_bbox(landmarks)
    xmin, ymin, xmax, _ = bbox
    scale = FACE_CROP_SIZE / float(xmax - xmin)

    points = landmarks[config.start:config.end].astype(np.float32).copy()
    points[:, 0] = (points[:, 0] - xmin) * scale - FACE_BORDER
    points[:, 1] = (points[:, 1] - ymin) * scale - FACE_BORDER
    return points


def mouth_roi_from_landmarks(landmarks: np.ndarray, config: MouthRoiConfig) -> tuple[int, int, int, int]:
    points = _project_mouth_points_to_inner(landmarks, config)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max((x2 - x1 + 2 * config.pad) * config.expand_x, float(config.min_w))
    height = max((y2 - y1 + 2 * config.pad) * config.expand_y, float(config.min_h))

    rx1 = int(round(cx - width / 2.0))
    rx2 = int(round(cx + width / 2.0))
    ry1 = int(round(cy - height / 2.0))
    ry2 = int(round(cy + height / 2.0))

    rx1 = max(0, min(FACE_INNER_SIZE - 1, rx1))
    ry1 = max(0, min(FACE_INNER_SIZE - 1, ry1))
    rx2 = max(rx1 + 1, min(FACE_INNER_SIZE, rx2))
    ry2 = max(ry1 + 1, min(FACE_INNER_SIZE, ry2))
    return rx1, ry1, rx2, ry2


def mouth_mask_from_landmarks(landmarks: np.ndarray, config: MouthRoiConfig) -> torch.Tensor:
    x1, y1, x2, y2 = mouth_roi_from_landmarks(landmarks, config)
    mask = np.zeros((1, FACE_INNER_SIZE, FACE_INNER_SIZE), dtype=np.float32)
    mask[:, y1:y2, x1:x2] = 1.0
    return torch.from_numpy(mask)


class MouthRoiDataset(Dataset):
    def __init__(self, dataset_dir: str, mode: str, mouth_config: MouthRoiConfig | None = None):
        if mode not in _AUDIO_FEAT_FILE:
            raise ValueError(f"Unknown asr mode: {mode}")
        self.mode = mode
        self.mouth_config = mouth_config or MouthRoiConfig()

        full_body_dir = os.path.join(dataset_dir, "full_body_img")
        landmarks_dir = os.path.join(dataset_dir, "landmarks")
        n_frames = count_jpgs(full_body_dir)
        self.img_path_list = [os.path.join(full_body_dir, f"{i}.jpg") for i in range(n_frames)]
        self.lms_path_list = [os.path.join(landmarks_dir, f"{i}.lms") for i in range(n_frames)]

        audio_feats_path = os.path.join(dataset_dir, _AUDIO_FEAT_FILE[mode])
        self.audio_feats = np.load(audio_feats_path).astype(np.float32)

    def __len__(self) -> int:
        return min(self.audio_feats.shape[0], len(self.img_path_list))

    def _build_target_masked_and_mouth(self, idx: int):
        face_crop = load_face_crop(self.img_path_list[idx], self.lms_path_list[idx])
        inner = extract_inner(face_crop)
        target_T = hwc_to_chw_tensor(inner.copy())
        masked_T = hwc_to_chw_tensor(mask_mouth(inner))
        landmarks = read_landmarks(self.lms_path_list[idx])
        mouth_mask_T = mouth_mask_from_landmarks(landmarks, self.mouth_config)
        return target_T, masked_T, mouth_mask_T

    def _build_reference(self) -> torch.Tensor:
        ref_idx = random.randint(0, len(self) - 1)
        return self._build_reference_at(ref_idx)

    def _build_reference_at(self, ref_idx: int) -> torch.Tensor:
        face_crop = load_face_crop(self.img_path_list[ref_idx], self.lms_path_list[ref_idx])
        return hwc_to_chw_tensor(extract_inner(face_crop))

    def _build_frame(self, idx: int, ref_T: torch.Tensor | None = None):
        target_T, masked_T, mouth_mask_T = self._build_target_masked_and_mouth(idx)
        ref_T = ref_T if ref_T is not None else self._build_reference()
        img_concat_T = torch.cat([ref_T, masked_T], dim=0)

        audio_feat = gather_audio_window(self.audio_feats, idx)
        audio_feat = reshape_audio_feat(audio_feat, self.mode)

        return img_concat_T, target_T, audio_feat, mouth_mask_T

    def __getitem__(self, idx: int):
        return self._build_frame(idx)


class TemporalMouthRoiDataset(MouthRoiDataset):
    """Return adjacent frame pairs with one shared reference frame.

    Sharing the reference keeps the temporal delta loss focused on mouth motion
    instead of random reference-frame differences.
    """

    def __init__(
        self,
        dataset_dir: str,
        mode: str,
        mouth_config: MouthRoiConfig | None = None,
        temporal_stride: int = 1,
    ):
        super().__init__(dataset_dir, mode, mouth_config=mouth_config)
        if temporal_stride < 1:
            raise ValueError("temporal_stride must be >= 1")
        self.temporal_stride = temporal_stride

    def __len__(self) -> int:
        base_len = MouthRoiDataset.__len__(self)
        return max(0, base_len - self.temporal_stride)

    def __getitem__(self, idx: int):
        base_len = MouthRoiDataset.__len__(self)
        ref_T = self._build_reference_at(random.randint(0, base_len - 1))
        first = self._build_frame(idx, ref_T=ref_T)
        second = self._build_frame(idx + self.temporal_stride, ref_T=ref_T)
        return tuple(torch.stack(items, dim=0) for items in zip(first, second))
