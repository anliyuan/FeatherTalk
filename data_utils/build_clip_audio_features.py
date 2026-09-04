"""Build frame-aligned audio features for a dataset made from independent clips.

Each source clip is decoded and aligned independently so AAC priming/padding
cannot accumulate across concatenation points. The output also records clip
boundaries, allowing the training dataset to zero-pad audio context at cuts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feather_hubert.feather_hubert import (  # noqa: E402
    HUBERT_KERNEL,
    HUBERT_STRIDE,
    get_best_device,
    get_feather_hubert_from_16k_speech,
    load_feather_hubert,
)


SAMPLE_RATE = 16000
SAMPLES_PER_VIDEO_FRAME = SAMPLE_RATE // 25


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("frame counts must be positive integers")
    return values


def probe_video_start(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return max(0.0, float(value)) if value and value != "N/A" else 0.0


def decode_aligned_audio(path: Path, video_start: float) -> np.ndarray:
    audio_filter = f"atrim=start={video_start:.9f},asetpts=PTS-STARTPTS"
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-af",
            audio_filter,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def fit_waveform_to_frames(waveform: np.ndarray, frame_count: int) -> np.ndarray:
    # +80 samples are required by the HuBERT-compatible valid-convolution
    # frontend to produce exactly two 20 ms tokens for every 40 ms video frame.
    target_samples = frame_count * SAMPLES_PER_VIDEO_FRAME + (
        HUBERT_KERNEL - HUBERT_STRIDE
    )
    if waveform.shape[0] >= target_samples:
        return waveform[:target_samples]
    return np.pad(waveform, (0, target_samples - waveform.shape[0]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build independent-clip FeatherHuBERT training features"
    )
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame_counts", type=parse_int_list, required=True)
    args = parser.parse_args()

    segments = sorted(args.segments_dir.glob("*.mp4"))
    if len(segments) != len(args.frame_counts):
        raise ValueError(
            f"found {len(segments)} clips but received "
            f"{len(args.frame_counts)} frame counts"
        )

    image_dir = args.dataset_dir / "full_body_img"
    image_count = len(list(image_dir.glob("*.jpg")))
    if sum(args.frame_counts) != image_count:
        raise ValueError(
            f"frame counts sum to {sum(args.frame_counts)}, "
            f"but the dataset contains {image_count} images"
        )

    decoded: list[np.ndarray] = []
    starts: list[float] = []
    for path in segments:
        video_start = probe_video_start(path)
        starts.append(video_start)
        decoded.append(decode_aligned_audio(path, video_start))

    # Match the normalization contract used by regular offline extraction while
    # still running the temporal encoder independently for every clip.
    all_samples = np.concatenate(decoded)
    mean = float(all_samples.mean())
    std = float(np.sqrt(all_samples.var() + 1e-7))

    device = get_best_device()
    model = load_feather_hubert(args.checkpoint, device=device)
    feature_parts: list[np.ndarray] = []
    ranges = []
    frame_start = 0
    for index, (path, waveform, frame_count, video_start) in enumerate(
        zip(segments, decoded, args.frame_counts, starts)
    ):
        fitted = fit_waveform_to_frames(waveform, frame_count)
        normalized = (fitted - mean) / std
        hidden = get_feather_hubert_from_16k_speech(
            normalized,
            model,
            device=device,
            normalize=False,
        )
        hidden = hidden[: frame_count * 2]
        if hidden.shape[0] != frame_count * 2:
            raise RuntimeError(
                f"clip {path.name} produced {hidden.shape[0]} tokens; "
                f"expected {frame_count * 2}"
            )
        part = hidden.reshape(frame_count, 2, model.config.output_dim).numpy()
        feature_parts.append(part)
        frame_end = frame_start + frame_count
        ranges.append(
            {
                "clip_index": index,
                "clip": path.name,
                "start": frame_start,
                "end": frame_end,
                "frame_count": frame_count,
                "video_start_seconds": video_start,
                "decoded_samples": int(waveform.shape[0]),
                "fitted_samples": int(fitted.shape[0]),
            }
        )
        print(
            f"[{index + 1}/{len(segments)}] {path.name}: "
            f"frames={frame_count}, features={part.shape[0]}"
        )
        frame_start = frame_end

    features = np.concatenate(feature_parts).astype(np.float32, copy=False)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.dataset_dir / "aud_hu.npy", features)
    metadata = {
        "version": 1,
        "fps": 25,
        "audio_sample_rate": SAMPLE_RATE,
        "feature_shape": list(features.shape),
        "normalization_mean": mean,
        "normalization_std": std,
        "ranges": ranges,
    }
    with open(args.dataset_dir / "clip_ranges.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"wrote {args.dataset_dir / 'aud_hu.npy'} {features.shape}")
    print(f"wrote {args.dataset_dir / 'clip_ranges.json'}")


if __name__ == "__main__":
    main()
