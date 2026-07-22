"""Streaming inference demo for FeatherTalk.

This file keeps the original Wenet ONNX streaming path and adds a HuBERT-shape
path for FeatherHuBERT.  The two modes differ only on the audio encoder side
and the UNet audio input shape:

* wenet: fbank -> Wenet encoder ONNX -> audio shape [1, 128, 16, 32]
* feather_hubert: waveform -> FeatherHuBERT -> audio shape [1, 16, 32, 32]

Input audio is still consumed as 10 ms / 16 kHz PCM chunks.  The Python
FeatherHuBERT path recomputes features from the current utterance buffer, so it
is meant as a correctness/reference streaming demo.  For production/mobile use,
export FeatherHuBERT to ONNX and pass it through --encoder_onnx.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime

try:
    import kaldi_native_fbank as knf
except ImportError:  # FeatherHuBERT mode does not need fbank.
    knf = None


SAMPLE_RATE = 16000
FRAME_LEN = 160  # 10 ms @ 16 kHz

ASR_WENET = "wenet"
ASR_FEATHER = "feather_hubert"
ASR_CHOICES = (ASR_WENET, ASR_FEATHER, "hubert")
ASR_ALIASES = {"hubert": ASR_FEATHER}

FPS_BY_ASR = {
    ASR_WENET: 20,
    ASR_FEATHER: 25,
}
IDLE_LOOP_BY_ASR = {
    ASR_WENET: 5,
    ASR_FEATHER: 4,
}

WENET_TRIGGER_LEN = 11040  # 690 ms
WENET_CHUNK_DROP = 800  # 50 ms
MEL_BINS = 80
WENET_FEAT_FRAMES = 67
PRE_AUDIO_LEN = 32 * FRAME_LEN  # 320 ms silence prefix for Wenet encoder
PLAY_PRE_PAD = 13440

FEATHER_TOKENS_PER_VIDEO_FRAME = 2
FEATHER_DEFAULT_OUTPUT_DIM = 1024
FEATHER_RIGHT_CONTEXT_FRAMES = 4

SILENCE_THRESHOLD = 100
UNET_FEAT_WINDOW = 8
USING_FEAT_INIT = 4

_FBANK_OPTS = None


def parse_args():
    parser = argparse.ArgumentParser(description="FeatherTalk streaming inference demo")
    parser.add_argument("--data_path", type=str, default="./dataset_kanghui_wenet/111/",
                        help="Dataset dir. Supports img_inference/lms_inference or full_body_img/landmarks.")
    parser.add_argument("--audio_wav", type=str, default="1.wav", help="Input test wav")
    parser.add_argument("--out_video", type=str, default="./test_video.mp4")
    parser.add_argument("--out_audio", type=str, default="./test_audio.wav")
    parser.add_argument("--video_size", type=int, nargs=2, default=[1280, 720],
                        help="Output video resolution: W H")
    parser.add_argument("--asr", type=str, default=ASR_WENET, choices=ASR_CHOICES,
                        help="wenet, or feather_hubert. hubert is an alias of feather_hubert here.")
    parser.add_argument("--unet_onnx", type=str, default="",
                        help="UNet ONNX path. Defaults to <data_path>/unet.onnx")
    parser.add_argument("--encoder_onnx", type=str, default="",
                        help="Wenet encoder ONNX, or optional FeatherHuBERT ONNX.")
    parser.add_argument("--feather_hubert_checkpoint", type=str, default="",
                        help="FeatherHuBERT .pth checkpoint when --asr feather_hubert and no --encoder_onnx is used.")
    parser.add_argument("--feather_right_context_frames", type=int, default=FEATHER_RIGHT_CONTEXT_FRAMES,
                        help="Delay FeatherHuBERT feature emission by N video frames for steadier streaming features.")
    parser.add_argument("--fps", type=float, default=0,
                        help="Override output fps. Defaults to 20 for Wenet and 25 for FeatherHuBERT.")
    parser.add_argument("--cpu_onnx", action="store_true",
                        help="Force ONNX Runtime CPUExecutionProvider.")
    return parser.parse_args()


def _normalize_asr(asr: str) -> str:
    return ASR_ALIASES.get(asr, asr)


def _get_fbank_opts():
    global _FBANK_OPTS
    if knf is None:
        raise RuntimeError(
            "kaldi_native_fbank is required for --asr wenet. "
            "Install it or use --asr feather_hubert."
        )
    if _FBANK_OPTS is None:
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = MEL_BINS
        opts.mel_opts.debug_mel = False
        _FBANK_OPTS = opts
    return _FBANK_OPTS


def _ort_providers(force_cpu: bool = False) -> List[str]:
    if force_cpu:
        return ["CPUExecutionProvider"]
    available = set(onnxruntime.get_available_providers())
    providers: List[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _read_landmarks_to_bbox(lms_path: str) -> Tuple[int, int, int, int]:
    pts = []
    with open(lms_path, "r") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if line:
                pts.append(np.fromstring(line, sep=" ", dtype=np.float32))
    lms = np.array(pts, dtype=np.int32)
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    ymax = ymin + (xmax - xmin)
    return xmin, ymin, xmax, ymax


def _build_unet_inputs(crop_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    crop_ori = crop_img.copy()
    inner = crop_img[4:164, 4:164].copy()
    masked = cv2.rectangle(inner.copy(), (5, 5), (150, 145), (0, 0, 0), -1)

    masked = masked.transpose(2, 0, 1).astype(np.float32) / 255.0
    inner = inner.transpose(2, 0, 1).astype(np.float32) / 255.0
    onnx_in = np.concatenate(
        (np.expand_dims(inner, 0), np.expand_dims(masked, 0)),
        axis=1,
    )
    return onnx_in, crop_ori


def _numeric_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def _sorted_files(directory: str, suffix: str) -> List[Path]:
    root = Path(directory)
    return sorted(
        [path for path in root.iterdir() if path.suffix.lower() == suffix],
        key=_numeric_sort_key,
    )


def _resolve_asset_dirs(data_path: str) -> Tuple[str, str]:
    candidates = [
        ("img_inference", "lms_inference"),
        ("full_body_img", "landmarks"),
    ]
    for img_name, lms_name in candidates:
        img_dir = os.path.join(data_path, img_name)
        lms_dir = os.path.join(data_path, lms_name)
        if os.path.isdir(img_dir) and os.path.isdir(lms_dir):
            return img_dir, lms_dir
    raise FileNotFoundError(
        f"{data_path} must contain img_inference/lms_inference or full_body_img/landmarks."
    )


def _resolve_default_path(path: str, data_path: str, name: str) -> str:
    return path or os.path.join(data_path, name)


class _BounceIndex:
    """0,1,...,N-1,N-2,...,1,0,... index generator."""

    def __init__(self, n_frames: int):
        if n_frames < 2:
            raise ValueError("At least 2 frames are required for streaming inference.")
        self.n_frames = n_frames
        self.index = 0
        self.step = 1

    def advance(self):
        self.index += self.step
        if self.index >= self.n_frames - 1:
            self.step = -1
        elif self.index <= 0:
            self.step = 1


class DiHumanProcessor:
    def __init__(
        self,
        data_path: str,
        asr: str = ASR_WENET,
        unet_onnx: str = "",
        encoder_onnx: str = "",
        feather_hubert_checkpoint: str = "",
        feather_right_context_frames: int = FEATHER_RIGHT_CONTEXT_FRAMES,
        force_cpu_onnx: bool = False,
    ):
        self.data_path = data_path
        self.asr = _normalize_asr(asr)
        self.idle_loop = IDLE_LOOP_BY_ASR[self.asr]
        self.providers = _ort_providers(force_cpu_onnx)
        self.feather_right_context_frames = max(0, int(feather_right_context_frames))

        full_body_img_dir, lms_dir = _resolve_asset_dirs(data_path)
        self.full_body_img_list, self.bbox_list = self._load_assets(full_body_img_dir, lms_dir)
        self.frame_picker = _BounceIndex(len(self.bbox_list))

        unet_path = _resolve_default_path(unet_onnx, data_path, "unet.onnx")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet ONNX not found: {unet_path}")
        self.ort_unet = onnxruntime.InferenceSession(unet_path, providers=self.providers)

        self.ort_ae: Optional[onnxruntime.InferenceSession] = None
        self.feather_model: Any = None
        self.feather_device: Any = None
        self.feather_output_dim = FEATHER_DEFAULT_OUTPUT_DIM
        self._torch = None
        self._feather_normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._feather_expected_frames: Optional[Callable[[int], int]] = None
        self._feather_make_even: Optional[Callable[[Any], Any]] = None

        if self.asr == ASR_WENET:
            self._init_wenet_encoder(_resolve_default_path(encoder_onnx, data_path, "encoder.onnx"))
        else:
            self._init_feather_encoder(encoder_onnx, feather_hubert_checkpoint)

        self.counter = 0
        self.empty_audio_counter = 56
        self.is_processing = False
        self.silence = True
        self._reset_audio_state()

    def _load_assets(self, img_dir: str, lms_dir: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        img_paths = _sorted_files(img_dir, ".jpg")
        if not img_paths:
            img_paths = _sorted_files(img_dir, ".png")
        lms_paths = _sorted_files(lms_dir, ".lms")

        n_available = min(len(img_paths), len(lms_paths))
        if n_available < 2:
            raise FileNotFoundError(f"Need at least 2 image/landmark pairs under {img_dir} and {lms_dir}.")

        # The legacy preprocessing often leaves the final landmark frame less reliable.
        n_frames = n_available - 1 if n_available > 2 else n_available
        images: List[np.ndarray] = []
        boxes: List[Tuple[int, int, int, int]] = []
        for img_path, lms_path in zip(img_paths[:n_frames], lms_paths[:n_frames]):
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            images.append(img)
            boxes.append(_read_landmarks_to_bbox(str(lms_path)))
        return images, boxes

    def _init_wenet_encoder(self, encoder_path: str):
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Wenet encoder ONNX not found: {encoder_path}")
        self.ort_ae = onnxruntime.InferenceSession(encoder_path, providers=self.providers)
        self._reset_wenet_cache()

    def _init_feather_encoder(self, encoder_onnx: str, checkpoint: str):
        if encoder_onnx:
            if not os.path.exists(encoder_onnx):
                raise FileNotFoundError(f"FeatherHuBERT ONNX not found: {encoder_onnx}")
            self.ort_ae = onnxruntime.InferenceSession(encoder_onnx, providers=self.providers)
            return

        if not checkpoint:
            raise ValueError("--feather_hubert_checkpoint is required for --asr feather_hubert without --encoder_onnx.")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"FeatherHuBERT checkpoint not found: {checkpoint}")

        import torch
        from data_utils.feather_hubert.feather_hubert import (
            expected_hubert_frames,
            get_best_device,
            load_feather_hubert,
            make_even_first_dim,
            normalize_waveform,
        )

        self._torch = torch
        self.feather_device = get_best_device()
        self.feather_model = load_feather_hubert(checkpoint, device=self.feather_device)
        self.feather_output_dim = int(self.feather_model.config.output_dim)
        if self.feather_output_dim != FEATHER_DEFAULT_OUTPUT_DIM:
            raise ValueError(
                f"FeatherHuBERT output_dim={self.feather_output_dim}, but the released UNet expects 1024."
            )
        self._feather_expected_frames = expected_hubert_frames
        self._feather_make_even = make_even_first_dim
        self._feather_normalize = normalize_waveform

    def _reset_wenet_cache(self):
        self.offset = np.ones((1,), dtype=np.int64) * 100
        self.att_cache = np.zeros([3, 8, 16, 128], dtype=np.float32)
        self.cnn_cache = np.zeros([3, 1, 512, 14], dtype=np.float32)

    def _reset_audio_state(self):
        if self.asr == ASR_WENET:
            self._reset_wenet_cache()
            self.audio_play_list: List[int] = [0] * PLAY_PRE_PAD
            self.audio_queue_get_feat = np.zeros([PRE_AUDIO_LEN], dtype=np.int16)
            self.using_feat = np.zeros([USING_FEAT_INIT, 16, 512], dtype=np.float32)
        else:
            self.audio_play_list = []
            self.audio_queue_get_feat = np.zeros([0], dtype=np.int16)
            self.using_feat = np.zeros([USING_FEAT_INIT, 2, self.feather_output_dim], dtype=np.float32)
            self.feather_audio_buffer = np.zeros([0], dtype=np.float32)
            self.feather_emitted_frames = 0
            self.pending_feature_frames = 0

    def reset(self):
        self._reset_audio_state()
        self.counter = 0
        self.is_processing = True

    def _detect_silence(self, audio_frame: np.ndarray):
        if not np.any(audio_frame):
            if not self.silence:
                self.empty_audio_counter += 1
            if self.empty_audio_counter >= SILENCE_THRESHOLD:
                self.silence = True
        else:
            self.empty_audio_counter = 0
            self.silence = False

    def _next_idle_img(self) -> Tuple[Optional[np.ndarray], int]:
        if self.counter == 0:
            img = self.full_body_img_list[self.frame_picker.index].copy()
            self.frame_picker.advance()
            self.counter = 1
            return img, 1
        self.counter += 1
        if self.counter == self.idle_loop:
            self.counter = 0
        return None, 0

    def _pop_play_audio(self) -> np.ndarray:
        if self.audio_play_list:
            audio = np.array(self.audio_play_list[:FRAME_LEN], dtype=np.int16)
            self.audio_play_list = self.audio_play_list[FRAME_LEN:]
            if audio.shape[0] < FRAME_LEN:
                audio = np.pad(audio, (0, FRAME_LEN - audio.shape[0]))
            return audio
        return np.zeros([FRAME_LEN], dtype=np.int16)

    def _run_wenet_encoder(self) -> np.ndarray:
        if self.ort_ae is None:
            raise RuntimeError("Wenet encoder is not initialized.")
        fbank = knf.OnlineFbank(_get_fbank_opts())
        fbank.accept_waveform(SAMPLE_RATE, self.audio_queue_get_feat.tolist())
        self.audio_play_list.extend(
            self.audio_queue_get_feat[PRE_AUDIO_LEN:PRE_AUDIO_LEN + WENET_CHUNK_DROP].tolist()
        )

        mel = np.array([[fbank.get_frame(i) for i in range(fbank.num_frames_ready)]])
        mel = mel[:, :, :WENET_FEAT_FRAMES, :]
        inputs = {
            "chunk": mel.astype(np.float32),
            "offset": self.offset,
            "att_cache": self.att_cache.astype(np.float32),
            "cnn_cache": self.cnn_cache.astype(np.float32),
        }
        outs = self.ort_ae.run(None, inputs)
        return outs[0].astype(np.float32)

    def _run_feather_encoder(self) -> np.ndarray:
        if self._feather_expected_frames is None:
            from data_utils.feather_hubert.feather_hubert import expected_hubert_frames

            self._feather_expected_frames = expected_hubert_frames

        expected_tokens = self._feather_expected_frames(int(self.feather_audio_buffer.shape[0]))
        expected_video_frames = expected_tokens // FEATHER_TOKENS_PER_VIDEO_FRAME
        if expected_video_frames <= self.feather_emitted_frames:
            return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)

        if self.ort_ae is not None:
            speech = self.feather_audio_buffer.astype(np.float32)
            speech = (speech - speech.mean()) / np.sqrt(speech.var() + 1e-7)
            encoder_input = speech[None, :]
            input_name = self.ort_ae.get_inputs()[0].name
            hidden = self.ort_ae.run(None, {input_name: encoder_input})[0][0].astype(np.float32)
            if hidden.shape[0] % 2 == 1:
                hidden = hidden[:-1]
        else:
            if self.feather_model is None or self._torch is None:
                raise RuntimeError("FeatherHuBERT checkpoint model is not initialized.")
            speech = self._feather_normalize(self.feather_audio_buffer)
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(speech).to(self.feather_device)[None]
                hidden_tensor = self.feather_model(tensor)[0].detach().cpu()
                hidden = self._feather_make_even(hidden_tensor).numpy().astype(np.float32)

        if hidden.shape[0] < FEATHER_TOKENS_PER_VIDEO_FRAME:
            return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)

        video_features = hidden.reshape(-1, FEATHER_TOKENS_PER_VIDEO_FRAME, hidden.shape[-1])
        stable_frames = max(0, video_features.shape[0] - self.feather_right_context_frames)
        if stable_frames <= self.feather_emitted_frames:
            return np.empty((0, 2, self.feather_output_dim), dtype=np.float32)

        new_features = video_features[self.feather_emitted_frames:stable_frames]
        self.feather_emitted_frames = stable_frames
        return new_features.astype(np.float32)

    def _audio_feat_for_unet(self) -> np.ndarray:
        if self.asr == ASR_WENET:
            return self.using_feat.reshape(1, 128, 16, 32)
        return self.using_feat.reshape(1, 16, 32, 32)

    def _run_unet(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        xmin, ymin, xmax, ymax = bbox
        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (168, 168))
        onnx_in, crop_ori = _build_unet_inputs(crop_img)

        audio_feat = self._audio_feat_for_unet()
        inputs = {
            self.ort_unet.get_inputs()[0].name: onnx_in,
            self.ort_unet.get_inputs()[1].name: audio_feat,
        }
        outs = self.ort_unet.run(None, inputs)
        pred = (outs[0][0].transpose(1, 2, 0) * 255).astype(np.uint8)

        crop_ori[4:164, 4:164] = pred
        crop_ori = cv2.resize(crop_ori, (w, h))
        img[ymin:ymax, xmin:xmax] = crop_ori
        return img

    def _process_wenet(self, audio_frame: np.ndarray):
        self.audio_queue_get_feat = np.concatenate([self.audio_queue_get_feat, audio_frame], axis=0)
        if self.audio_queue_get_feat.shape[0] >= WENET_TRIGGER_LEN:
            audio_feat = self._run_wenet_encoder()
            self.audio_queue_get_feat = self.audio_queue_get_feat[WENET_CHUNK_DROP:]
            self.using_feat = np.concatenate([self.using_feat, audio_feat], axis=0)

            img = self.full_body_img_list[self.frame_picker.index].copy()
            bbox = self.bbox_list[self.frame_picker.index]
            self.frame_picker.advance()
            if self.using_feat.shape[0] >= UNET_FEAT_WINDOW:
                img = self._run_unet(img, bbox)
                self.using_feat = self.using_feat[1:]
            self.counter = 1
            return img.copy(), self._pop_play_audio(), 1

        return_img, check_img = self._next_idle_img()
        return return_img, self._pop_play_audio(), check_img

    def _process_feather(self, audio_frame: np.ndarray):
        self.audio_play_list.extend(audio_frame.tolist())
        audio_float = audio_frame.astype(np.float32) / 32768.0
        self.feather_audio_buffer = np.concatenate([self.feather_audio_buffer, audio_float], axis=0)

        new_features = self._run_feather_encoder()
        if new_features.shape[0] > 0:
            self.using_feat = np.concatenate([self.using_feat, new_features], axis=0)
            self.pending_feature_frames += int(new_features.shape[0])

        if self.pending_feature_frames > 0 and self.using_feat.shape[0] >= UNET_FEAT_WINDOW:
            img = self.full_body_img_list[self.frame_picker.index].copy()
            bbox = self.bbox_list[self.frame_picker.index]
            self.frame_picker.advance()
            img = self._run_unet(img, bbox)
            self.using_feat = self.using_feat[1:]
            self.pending_feature_frames -= 1
            self.counter = 1
            return img.copy(), self._pop_play_audio(), 1

        return_img, check_img = self._next_idle_img()
        return return_img, self._pop_play_audio(), check_img

    def process(self, audio_frame: np.ndarray):
        audio_frame = audio_frame.astype(np.int16)
        if audio_frame.shape[0] < FRAME_LEN:
            audio_frame = np.pad(audio_frame, (0, FRAME_LEN - audio_frame.shape[0]))
        elif audio_frame.shape[0] > FRAME_LEN:
            audio_frame = audio_frame[:FRAME_LEN]

        self._detect_silence(audio_frame)
        if self.silence:
            if self.is_processing:
                self._reset_audio_state()
            self.is_processing = False
            return_img, check_img = self._next_idle_img()
            return return_img, np.zeros([FRAME_LEN], dtype=np.int16), check_img

        if not self.is_processing:
            self.reset()

        if self.asr == ASR_WENET:
            return self._process_wenet(audio_frame)
        return self._process_feather(audio_frame)


def _select_fourcc(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc("M", "J", "P", "G")
    return cv2.VideoWriter_fourcc(*"mp4v")


def _resize_for_writer(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h))


def main(arg):
    import soundfile as sf
    from scipy.io import wavfile

    asr = _normalize_asr(arg.asr)
    stream, sr = sf.read(arg.audio_wav)
    if stream.ndim == 2:
        stream = stream[:, 0]
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz wav, got {sr} Hz: {arg.audio_wav}")
    stream = (stream.astype(np.float32) * 32767).astype(np.int16)

    w, h = arg.video_size
    fps = arg.fps if arg.fps > 0 else FPS_BY_ASR[asr]
    writer = cv2.VideoWriter(arg.out_video, _select_fourcc(arg.out_video), fps, (w, h))
    processor = DiHumanProcessor(
        arg.data_path,
        asr=asr,
        unet_onnx=arg.unet_onnx,
        encoder_onnx=arg.encoder_onnx,
        feather_hubert_checkpoint=arg.feather_hubert_checkpoint,
        feather_right_context_frames=arg.feather_right_context_frames,
        force_cpu_onnx=arg.cpu_onnx,
    )

    audio_out: List[np.ndarray] = []
    n_chunks = math.ceil(stream.shape[0] / FRAME_LEN)
    for i in range(n_chunks):
        a = i * FRAME_LEN
        b = min(a + FRAME_LEN, stream.shape[0])
        audio_frame = stream[a:b]
        img, playing_audio, check_img = processor.process(audio_frame)
        audio_out.append(playing_audio)
        if check_img and img is not None:
            writer.write(_resize_for_writer(img, (w, h)))

    writer.release()
    audio_data = np.concatenate(audio_out).astype(np.int16)
    wavfile.write(arg.out_audio, SAMPLE_RATE, audio_data)
    print(f"[done] asr={asr}, fps={fps}, video saved to {arg.out_video}, audio saved to {arg.out_audio}")


if __name__ == "__main__":
    main(parse_args())


# Merge audio and video:
# ffmpeg -i test_video.mp4 -i test_audio.wav -c:v libx264 -c:a aac result_test.mp4
