"""FeatherTalk 144x144 audio-conditioned UNet."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SIZE = 144
AUDIO_TOKENS = 40
AUDIO_FEATURE_DIM = 1024


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        use_res_connect: bool,
        expand_ratio: int = 6,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.use_res_connect = use_res_connect
        hidden_dim = inp * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        return x + output if self.use_res_connect else output


class DoubleConvDW(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride=stride,
                use_res_connect=False,
                expand_ratio=2,
            ),
            InvertedResidual(
                out_channels,
                out_channels,
                stride=1,
                use_res_connect=True,
                expand_ratio=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class InConvDw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inconv = InvertedResidual(
            in_channels,
            out_channels,
            stride=1,
            use_res_connect=False,
            expand_ratio=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = DoubleConvDW(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([x1, x2], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalConvBlock(nn.Module):
    """Residual depthwise-separable temporal block over ``[B, C, T]``."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.point_bn = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.bn(self.depthwise(x)))
        x = self.point_bn(self.pointwise(x))
        return self.activation(x + residual)


class AudioAdapter(nn.Module):
    """Convert 40 HuBERT-compatible tokens into 20 visual-frame steps."""

    input_tokens = AUDIO_TOKENS
    output_steps = 20
    input_dim = AUDIO_FEATURE_DIM
    output_dim = 512

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(self.input_dim * 2)
        self.input_projection = nn.Conv1d(
            self.input_dim * 2, hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            TemporalConvBlock(hidden_dim, dilation=1),
            TemporalConvBlock(hidden_dim, dilation=2),
            TemporalConvBlock(hidden_dim, dilation=4),
            TemporalConvBlock(hidden_dim, dilation=8),
        )
        self.output_projection = nn.Conv1d(
            hidden_dim, self.output_dim, kernel_size=1
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[1:] != (
            self.input_tokens,
            self.input_dim,
        ):
            raise ValueError(
                f"audio must have shape [B, {self.input_tokens}, {self.input_dim}], "
                f"got {tuple(audio.shape)}"
            )
        batch = audio.shape[0]
        x = audio.reshape(batch, self.output_steps, self.input_dim * 2)
        x = self.norm(x).transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.output_projection(x)
        return x.transpose(1, 2).contiguous()


class CenteredAudioContext(nn.Module):
    """Combine the current audio step with a learned local context."""

    def __init__(self, audio_dim: int, center_index: int = 10, radius: int = 2):
        super().__init__()
        self.audio_dim = audio_dim
        self.center_index = center_index
        self.radius = radius
        self.score = nn.Linear(audio_dim, 1)

    @property
    def output_dim(self) -> int:
        return self.audio_dim * 2

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[-1] != self.audio_dim:
            raise ValueError(
                f"expected audio [B, T, {self.audio_dim}], got {tuple(audio.shape)}"
            )
        center = min(self.center_index, audio.shape[1] - 1)
        start = max(0, center - self.radius)
        end = min(audio.shape[1], center + self.radius + 1)
        local = audio[:, start:end]
        weights = torch.softmax(self.score(local).squeeze(-1), dim=1)
        local_context = torch.sum(local * weights.unsqueeze(-1), dim=1)
        return torch.cat([audio[:, center], local_context], dim=-1)


class SpatialAudioFusion(nn.Module):
    """Use each visual location to attend over the aligned audio sequence."""

    def __init__(
        self,
        visual_channels: int,
        audio_dim: int = 512,
        attention_dim: int = 128,
        num_heads: int = 4,
        center_index: int = 10,
        max_audio_steps: int = 20,
    ):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.visual_norm = nn.LayerNorm(visual_channels)
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.query = nn.Linear(visual_channels, attention_dim)
        self.key = nn.Linear(audio_dim, attention_dim)
        self.value = nn.Linear(audio_dim, visual_channels)
        self.out = nn.Linear(visual_channels, visual_channels)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.center_index = center_index
        self.max_audio_steps = max_audio_steps
        self.audio_position = nn.Parameter(torch.zeros(max_audio_steps, audio_dim))
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, max_audio_steps))
        with torch.no_grad():
            positions = torch.arange(max_audio_steps, dtype=torch.float32)
            self.relative_bias.copy_(
                -0.12 * (positions - center_index).abs().unsqueeze(0)
            )
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 4 or audio.ndim != 3:
            raise ValueError("expected visual [B,C,H,W] and audio [B,T,D]")
        batch, channels, height, width = visual.shape
        if audio.shape[1] > self.max_audio_steps:
            raise ValueError("audio sequence is longer than the configured context")

        tokens = visual.flatten(2).transpose(1, 2)
        audio = audio + self.audio_position[: audio.shape[1]].unsqueeze(0)
        q = self.query(self.visual_norm(tokens))
        k = self.key(self.audio_norm(audio))
        v = self.value(self.audio_norm(audio))

        q = q.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(
            batch, -1, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        logits = logits + self.relative_bias[:, : audio.shape[1]].view(
            1, self.num_heads, 1, audio.shape[1]
        )
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch, -1, channels)
        context = self.out(context)
        fused = tokens + torch.sigmoid(self.residual_gate) * context
        return fused.transpose(1, 2).reshape(batch, channels, height, width)


class Model(nn.Module):
    """Current FeatherTalk visual model.

    Inputs:
        image: ``[B, 6, 144, 144]``
        audio: ``[B, 40, 1024]``
    Output:
        generated face patch ``[B, 3, 144, 144]``
    """

    image_size = IMAGE_SIZE
    audio_half_window = 10
    raw_hubert_audio = True
    visual_channels = (24, 32, 64, 128, 256)
    audio_dim = 512
    center_audio_index = 10

    def __init__(self, n_channels: int = 6, mode: str = "hubert"):
        super().__init__()
        if mode != "hubert":
            raise ValueError("FeatherTalk requires HuBERT-compatible audio features")

        x1_ch, x2_ch, x3_ch, x4_ch, x5_ch = self.visual_channels
        self.n_channels = n_channels
        self.inc = InConvDw(n_channels, x1_ch)
        self.down1 = Down(x1_ch, x2_ch)
        self.down2 = Down(x2_ch, x3_ch)
        self.down3 = Down(x3_ch, x4_ch)
        self.down4 = Down(x4_ch, x5_ch)

        self.audio_model = AudioAdapter(hidden_dim=512)
        self.spatial_x3 = SpatialAudioFusion(x3_ch, self.audio_dim)
        self.spatial_x4 = SpatialAudioFusion(x4_ch, self.audio_dim)
        self.spatial_x5 = SpatialAudioFusion(x5_ch, self.audio_dim)
        self.audio_context = CenteredAudioContext(
            self.audio_dim, center_index=self.center_audio_index, radius=2
        )
        self.audio_to_x5 = nn.Linear(self.audio_context.output_dim, x5_ch)
        self.fuse_conv = DoubleConvDW(x5_ch * 2, x5_ch, stride=1)

        self.up1 = Up(x5_ch + x4_ch, x4_ch)
        self.up2 = Up(x4_ch + x3_ch, x3_ch)
        self.up3 = Up(x3_ch + x2_ch, x2_ch)
        self.up4 = Up(x2_ch + x1_ch, x1_ch)
        self.outc = OutConv(x1_ch, 3)

    def encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def decode_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        audio_sequence = self.audio_model(audio)
        x3 = self.spatial_x3(x3, audio_sequence)
        x4 = self.spatial_x4(x4, audio_sequence)
        x5 = self.spatial_x5(x5, audio_sequence)

        audio_context = self.audio_context(audio_sequence)
        audio_map = self.audio_to_x5(audio_context).unsqueeze(-1).unsqueeze(-1)
        audio_map = audio_map.expand(-1, -1, x5.shape[-2], x5.shape[-1])
        x5 = self.fuse_conv(torch.cat([x5, audio_map], dim=1))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.decode_features(*self.encode_image(image), audio)


def load_checkpoint_state(
    checkpoint_path: str, device: torch.device | str = "cpu"
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint
