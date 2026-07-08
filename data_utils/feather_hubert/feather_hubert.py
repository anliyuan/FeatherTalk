from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 16000
HUBERT_KERNEL = 400
HUBERT_STRIDE = 320


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def expected_hubert_frames(num_samples: int) -> int:
    if num_samples < HUBERT_KERNEL:
        return 0
    return (num_samples - (HUBERT_KERNEL - HUBERT_STRIDE)) // HUBERT_STRIDE


def make_even_first_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] % 2 == 1:
        return tensor[:-1]
    return tensor


def read_wav_16k(path: str | Path) -> np.ndarray:
    import soundfile as sf

    speech, sr = sf.read(path)
    if speech.ndim == 2:
        speech = speech[:, 0]
    speech = speech.astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa

        speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    return speech


def normalize_waveform(speech: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    speech = speech.astype(np.float32)
    return (speech - speech.mean()) / np.sqrt(speech.var() + eps)


def _pick_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.norm = nn.GroupNorm(_pick_group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class HubertStrideFrontend(nn.Module):
    """HuBERT-compatible valid Conv1d frontend.

    Kernels and strides match the HuBERT feature extractor so the output has
    one token every 320 samples, with the same frame count as HuBERT.
    """

    def __init__(self, channels: tuple[int, ...] = (64, 128, 256, 384, 512, 512, 512)):
        super().__init__()
        kernels = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        in_channels = 1
        layers = []
        for out_channels, kernel_size, stride in zip(channels, kernels, strides):
            layers.append(ConvNormAct1d(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform[:, None, :]
        elif waveform.ndim != 3:
            raise ValueError(f"Expected waveform shape [B, N] or [B, 1, N], got {tuple(waveform.shape)}")
        return self.layers(waveform)


class DepthwiseTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.05,
    ):
        super().__init__()
        hidden_channels = channels * expansion
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(_pick_group_count(channels), channels)
        self.pw_expand = nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
            bias=False,
        )
        self.act = nn.GELU()
        self.pw_project = nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(self.dw_conv(x))
        x = self.pw_project(x)
        return residual + self.dropout(x)


@dataclass(frozen=True)
class FeatherHuBERTConfig:
    channels: int = 512
    expansion: int = 2
    num_blocks: int = 12
    output_dim: int = 1024
    dropout: float = 0.05


def feather_hubert_config_from_mapping(mapping: dict | None) -> FeatherHuBERTConfig:
    if not mapping:
        return FeatherHuBERTConfig()
    defaults = asdict(FeatherHuBERTConfig())
    values = {}
    for key, default in defaults.items():
        value = mapping.get(key, default)
        values[key] = type(default)(value)
    return FeatherHuBERTConfig(**values)


class FeatherHuBERTEncoder(nn.Module):
    """FeatherHuBERT audio encoder.

    Input:
        16 kHz waveform, shape [B, samples] or [B, 1, samples].

    Output:
        Hidden features, shape [B, frames, 1024], where frames follow HuBERT's
        20 ms stride and valid-convolution frame count.
    """

    def __init__(self, config: FeatherHuBERTConfig | None = None):
        super().__init__()
        self.config = config or FeatherHuBERTConfig()
        self.frontend = HubertStrideFrontend(
            channels=(64, 128, 256, 384, self.config.channels, self.config.channels, self.config.channels)
        )
        dilations = [1, 2, 4, 8]
        self.encoder = nn.Sequential(
            *[
                DepthwiseTCNBlock(
                    channels=self.config.channels,
                    expansion=self.config.expansion,
                    dilation=dilations[index % len(dilations)],
                    dropout=self.config.dropout,
                )
                for index in range(self.config.num_blocks)
            ]
        )
        self.final_norm = nn.GroupNorm(_pick_group_count(self.config.channels), self.config.channels)
        self.proj = nn.Conv1d(self.config.channels, self.config.output_dim, kernel_size=1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        expected_frames = expected_hubert_frames(waveform.shape[-1])
        if expected_frames <= 0:
            raise ValueError(f"Waveform is too short for HuBERT-compatible output: {waveform.shape[-1]} samples")

        x = self.frontend(waveform)
        x = self.encoder(x)
        x = self.proj(F.gelu(self.final_norm(x)))
        x = x.transpose(1, 2).contiguous()

        if x.shape[1] < expected_frames:
            x = F.pad(x, (0, 0, 0, expected_frames - x.shape[1]))
        elif x.shape[1] > expected_frames:
            x = x[:, :expected_frames]
        return x


def load_feather_hubert(checkpoint: str | Path, device: torch.device | None = None) -> FeatherHuBERTEncoder:
    device = device or get_best_device()
    checkpoint_data = torch.load(checkpoint, map_location=device)
    config_data = checkpoint_data.get("config")
    if config_data is None:
        config_data = checkpoint_data.get("args") if isinstance(checkpoint_data, dict) else None
    model = FeatherHuBERTEncoder(feather_hubert_config_from_mapping(config_data))
    state_dict = checkpoint_data.get("model", checkpoint_data)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def get_feather_hubert_from_16k_speech(
    speech: np.ndarray,
    model: FeatherHuBERTEncoder,
    device: torch.device | None = None,
    chunk_samples: int = HUBERT_STRIDE * 1000,
    normalize: bool = True,
) -> torch.Tensor:
    device = device or get_best_device()
    model = model.to(device).eval()
    if speech.ndim == 2:
        speech = speech[:, 0]
    speech = speech.astype(np.float32)
    if normalize:
        speech = normalize_waveform(speech)
    total_expected = expected_hubert_frames(speech.shape[0])
    if total_expected <= 0:
        return torch.empty((0, model.config.output_dim), dtype=torch.float32)

    outputs = []
    num_iter = speech.shape[0] // chunk_samples
    for index in range(num_iter):
        if index == 0:
            start = 0
            end = chunk_samples - HUBERT_STRIDE + HUBERT_KERNEL
        else:
            start = chunk_samples * index
            end = start + (chunk_samples - HUBERT_STRIDE + HUBERT_KERNEL)
        chunk = torch.from_numpy(speech[start:end]).to(device)[None]
        if chunk.shape[-1] >= HUBERT_KERNEL:
            outputs.append(model(chunk)[0].cpu())

    tail = speech[chunk_samples * num_iter :]
    if tail.shape[0] >= HUBERT_KERNEL:
        outputs.append(model(torch.from_numpy(tail).to(device)[None])[0].cpu())

    hidden = torch.cat(outputs, dim=0) if outputs else torch.empty((0, model.config.output_dim))
    if hidden.shape[0] < total_expected:
        hidden = F.pad(hidden, (0, 0, 0, total_expected - hidden.shape[0]))
    else:
        hidden = hidden[:total_expected]
    return hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FeatherHuBERT feature extractor")
    parser.add_argument("--wav", type=str, default="", help="Input wav path")
    parser.add_argument("--checkpoint", type=str, default="", help="FeatherHuBERT checkpoint")
    parser.add_argument("--out", type=str, default="", help="Output .npy path. Defaults to *_feather_hu.npy")
    parser.add_argument("--stats", action="store_true", help="Print model parameter count")
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatherHuBERTConfig(
        channels=args.channels,
        expansion=args.expansion,
        num_blocks=args.num_blocks,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )
    model = FeatherHuBERTEncoder(config)
    if args.stats:
        params = count_parameters(model)
        print(f"parameters={params:,}")
        print(f"fp32_size_mb={params * 4 / 1024 / 1024:.2f}")
        print(f"int8_size_mb={params / 1024 / 1024:.2f}")

    if not args.wav:
        return
    if not args.checkpoint:
        raise ValueError("Please provide --checkpoint before extracting features.")

    device = get_best_device()
    model = load_feather_hubert(args.checkpoint, device=device)
    speech = read_wav_16k(args.wav)
    hidden = get_feather_hubert_from_16k_speech(speech, model, device=device)
    hidden = make_even_first_dim(hidden).reshape(-1, 2, model.config.output_dim)
    out_path = args.out or str(Path(args.wav).with_suffix("")) + "_feather_hu.npy"
    np.save(out_path, hidden.numpy())
    print(hidden.numpy().shape)
    print(out_path)


TinyHubertConfig = FeatherHuBERTConfig
TinyHubertLikeEncoder = FeatherHuBERTEncoder
tiny_hubert_config_from_mapping = feather_hubert_config_from_mapping
load_tiny_hubert = load_feather_hubert
get_tiny_hubert_from_16k_speech = get_feather_hubert_from_16k_speech


if __name__ == "__main__":
    main()
