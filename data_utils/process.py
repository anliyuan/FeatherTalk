"""一键预处理：从视频抽音频 / 抽帧 / 检测关键点 / 提取音频特征。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import cv2


HERE = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_FPS = 25


def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    print(f"[INFO] ===== extract audio: {video_path} -> {wav_path} =====")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-f", "wav",
        "-ar", str(sample_rate),
        wav_path,
    ]
    subprocess.run(cmd, check=True)
    print("[INFO] ===== extracted audio =====")


def _check_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps != _REQUIRED_FPS:
        raise ValueError(
            f"FeatherTalk requires a {_REQUIRED_FPS} FPS video (got {fps})"
        )
    return fps


def extract_images(video_path: str) -> None:
    _check_fps(video_path)

    base_dir = os.path.dirname(os.path.abspath(video_path))
    full_body_dir = os.path.join(base_dir, "full_body_img")
    os.makedirs(full_body_dir, exist_ok=True)

    print("extracting images...")
    cap = cv2.VideoCapture(video_path)
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(full_body_dir, f"{counter}.jpg"), frame)
        counter += 1
    cap.release()
    print(f"[INFO] extracted {counter} frames -> {full_body_dir}")


def get_audio_feature(wav_path: str, feather_hubert_checkpoint: str) -> None:
    print("extracting audio feature...")
    wav_path_abs = os.path.abspath(wav_path)
    out_path = os.path.join(os.path.dirname(wav_path_abs), "aud_hu.npy")
    cmd = [
        sys.executable,
        "feather_hubert/feather_hubert.py",
        "--wav",
        wav_path_abs,
        "--checkpoint",
        os.path.abspath(feather_hubert_checkpoint),
        "--out",
        out_path,
    ]
    # 强制切到 data_utils 目录，保证脚本里相对路径（conf/、scrfd onnx 等）能被解析
    subprocess.run(cmd, check=True, cwd=HERE)


def _sorted_jpgs(dir_path: str):
    def _key(name: str):
        stem = os.path.splitext(name)[0]
        return int(stem) if stem.isdigit() else name

    return sorted(
        (f for f in os.listdir(dir_path) if f.endswith(".jpg")),
        key=_key,
    )


def get_landmark(video_path: str, landmarks_dir: str) -> None:
    print("detecting landmarks...")
    base_dir = os.path.dirname(os.path.abspath(video_path))
    full_img_dir = os.path.join(base_dir, "full_body_img")

    # Landmark 类内部使用相对路径加载模型文件，必须在 data_utils 里跑
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        from get_landmark import Landmark
        landmark = Landmark()

        for img_name in _sorted_jpgs(full_img_dir):
            img_path = os.path.join(full_img_dir, img_name)
            lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
            pre_landmark, x1, y1 = landmark.detect(img_path)
            with open(lms_path, "w") as f:
                for p in pre_landmark:
                    f.write(f"{p[0] + x1} {p[1] + y1}\n")
    finally:
        os.chdir(cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to video file")
    parser.add_argument(
        "--feather_hubert_checkpoint",
        type=str,
        required=True,
        help="FeatherHuBERT checkpoint used to write aud_hu.npy",
    )
    opt = parser.parse_args()

    video_path = os.path.abspath(opt.path)
    base_dir = os.path.dirname(video_path)
    wav_path = os.path.join(base_dir, "aud.wav")
    landmarks_dir = os.path.join(base_dir, "landmarks")
    os.makedirs(landmarks_dir, exist_ok=True)

    extract_audio(video_path, wav_path)
    extract_images(video_path)
    get_landmark(video_path, landmarks_dir)
    get_audio_feature(wav_path, opt.feather_hubert_checkpoint)


if __name__ == "__main__":
    main()
