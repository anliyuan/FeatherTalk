"""Offline Python inference for the released FeatherTalk model."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np
import torch

from face_utils import (
    FACE_BORDER,
    FACE_CROP_SIZE,
    FACE_INNER_SIZE,
    compute_face_bbox,
    crop_face,
    extract_inner,
    gather_audio_window,
    hwc_to_chw_tensor,
    mask_mouth,
    mouth_soft_blend_mask,
    read_landmarks,
    reshape_audio_feat,
)
from model import Model, load_checkpoint_state


OUTPUT_FPS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FeatherTalk offline inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audio_feat", required=True, help="aud_hu.npy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument(
        "--audio_wav", default="", help="optional WAV merged into the output video"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> Model:
    model = Model().to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device))
    return model.eval()


def select_fourcc(path: str) -> int:
    if os.path.splitext(path)[1].lower() == ".avi":
        return cv2.VideoWriter_fourcc("M", "J", "P", "G")
    return cv2.VideoWriter_fourcc(*"mp4v")


def merge_audio(video_path: str, audio_path: str, output_path: str) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to merge audio")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ],
        check=True,
    )


class FramePicker:
    """Pick resource frames in a forward/backward loop."""

    def __init__(self, frame_count: int):
        if frame_count < 2:
            raise ValueError("at least two resource frames are required")
        self.frame_count = frame_count
        self.index = 0
        self.step = 0

    def next(self) -> int:
        if self.index >= self.frame_count - 1:
            self.step = -1
        if self.index <= 0:
            self.step = 1
        self.index += self.step
        return self.index


def prepare_model_input(
    image: np.ndarray, landmark_path: str, device: torch.device
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    bbox = compute_face_bbox(read_landmarks(landmark_path))
    xmin, ymin, xmax, ymax = bbox
    crop_height, crop_width = image[ymin:ymax, xmin:xmax].shape[:2]
    face_crop = crop_face(image, bbox, crop_size=FACE_CROP_SIZE)
    original_crop = face_crop.copy()
    inner = extract_inner(face_crop, inner_size=FACE_INNER_SIZE)

    reference = hwc_to_chw_tensor(inner.copy()).to(device)
    masked = hwc_to_chw_tensor(
        mask_mouth(inner.copy(), inner_size=FACE_INNER_SIZE)
    ).to(device)
    model_input = torch.cat([reference, masked], dim=0).unsqueeze(0)
    return model_input, original_crop, bbox, (crop_width, crop_height)


def paste_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    face_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    original_size: tuple[int, int],
) -> None:
    alpha = mouth_soft_blend_mask(FACE_INNER_SIZE).numpy()[0, :, :, None]
    current = face_crop[
        FACE_BORDER : FACE_BORDER + FACE_INNER_SIZE,
        FACE_BORDER : FACE_BORDER + FACE_INNER_SIZE,
    ].astype(np.float32)
    blended = alpha * prediction.astype(np.float32) + (1.0 - alpha) * current
    face_crop[
        FACE_BORDER : FACE_BORDER + FACE_INNER_SIZE,
        FACE_BORDER : FACE_BORDER + FACE_INNER_SIZE,
    ] = blended.clip(0, 255).round().astype(np.uint8)

    crop_width, crop_height = original_size
    face_crop = cv2.resize(face_crop, (crop_width, crop_height))
    xmin, ymin, xmax, ymax = bbox
    image[ymin:ymax, xmin:xmax] = face_crop


def run(args: argparse.Namespace, device: torch.device | None = None) -> None:
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    features = np.load(args.audio_feat).astype(np.float32)
    image_dir = os.path.join(args.dataset, "full_body_img")
    landmark_dir = os.path.join(args.dataset, "landmarks")
    frame_count = sum(name.endswith(".jpg") for name in os.listdir(image_dir))
    first_image = cv2.imread(os.path.join(image_dir, "0.jpg"))
    if first_image is None:
        raise FileNotFoundError("resource frame 0.jpg was not found")
    height, width = first_image.shape[:2]

    temporary_video = args.save_path
    if args.audio_wav:
        root, extension = os.path.splitext(args.save_path)
        temporary_video = f"{root}.silent{extension or '.mp4'}"
    writer = cv2.VideoWriter(
        temporary_video, select_fourcc(temporary_video), OUTPUT_FPS, (width, height)
    )

    model = load_model(args.checkpoint, device)
    picker = FramePicker(frame_count)
    with torch.no_grad():
        for frame_index in range(features.shape[0]):
            resource_index = picker.next()
            image = cv2.imread(os.path.join(image_dir, f"{resource_index}.jpg"))
            landmark_path = os.path.join(landmark_dir, f"{resource_index}.lms")
            model_input, face_crop, bbox, original_size = prepare_model_input(
                image, landmark_path, device
            )
            audio = reshape_audio_feat(
                gather_audio_window(features, frame_index)
            ).unsqueeze(0).to(device)
            prediction = model(model_input, audio)[0]
            prediction = (
                prediction.cpu().numpy().transpose(1, 2, 0) * 255.0
            ).clip(0, 255).astype(np.uint8)
            paste_prediction(image, prediction, face_crop, bbox, original_size)
            writer.write(image)
    writer.release()

    if args.audio_wav:
        merge_audio(temporary_video, args.audio_wav, args.save_path)
        os.remove(temporary_video)


if __name__ == "__main__":
    run(parse_args())
