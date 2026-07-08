from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.base_module import MobileOneBlock


_MAIN_CHANNELS = [32, 64, 128, 256, 512]


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int] = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileOneSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_conv_branches: int = 2,
        use_res_connect: bool = False,
    ):
        super().__init__()
        self.use_res_connect = use_res_connect and stride == 1 and in_channels == out_channels
        self.depthwise = MobileOneBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            num_conv_branches=num_conv_branches,
        )
        self.pointwise = MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            num_conv_branches=num_conv_branches,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise(self.depthwise(x))
        return x + out if self.use_res_connect else out


class DoubleConvMobileOne(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.double_conv = nn.Sequential(
            MobileOneSeparableBlock(in_channels, out_channels, stride=stride),
            MobileOneSeparableBlock(out_channels, out_channels, stride=1, use_res_connect=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class InConvMobileOne(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inconv = MobileOneSeparableBlock(in_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = DoubleConvMobileOne(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvMobileOne(in_channels, out_channels, stride=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x1, x2], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AudioConvWenet(nn.Module):
    def __init__(self, ch=_MAIN_CHANNELS):
        super().__init__()
        self.conv1 = MobileOneSeparableBlock(ch[2], ch[3], stride=1)
        self.conv2 = MobileOneSeparableBlock(ch[3], ch[3], stride=1, use_res_connect=True)
        self.conv3 = MobileOneBlock(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2), num_conv_branches=2)
        self.conv4 = MobileOneSeparableBlock(ch[3], ch[3], stride=1, use_res_connect=True)
        self.conv5 = ConvBNAct(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.conv6 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)
        self.conv7 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv7(x)


class AudioConvHubert(nn.Module):
    def __init__(self, ch=_MAIN_CHANNELS):
        super().__init__()
        self.conv1 = MobileOneSeparableBlock(16, ch[1], stride=1)
        self.conv2 = MobileOneSeparableBlock(ch[1], ch[2], stride=1)
        self.conv3 = MobileOneBlock(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2), num_conv_branches=2)
        self.conv4 = MobileOneSeparableBlock(ch[3], ch[3], stride=1, use_res_connect=True)
        self.conv5 = ConvBNAct(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.conv6 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)
        self.conv7 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv7(x)


_AUDIO_BRANCH = {
    "wenet": AudioConvWenet,
    "hubert": AudioConvHubert,
}


class Model(nn.Module):
    def __init__(self, n_channels: int = 6, mode: str = "wenet"):
        super().__init__()
        if mode not in _AUDIO_BRANCH:
            raise ValueError(f"Unknown asr mode: {mode}")

        ch = _MAIN_CHANNELS
        self.audio_model = _AUDIO_BRANCH[mode]()
        self.fuse_conv = nn.Sequential(
            DoubleConvMobileOne(ch[4] * 2, ch[4], stride=1),
            DoubleConvMobileOne(ch[4], ch[3], stride=1),
        )

        self.inc = InConvMobileOne(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.up1 = Up(ch[4], ch[3] // 2)
        self.up2 = Up(ch[3], ch[2] // 2)
        self.up3 = Up(ch[2], ch[1] // 2)
        self.up4 = Up(ch[1], ch[0])
        self.outc = OutConv(ch[0], 3)

    def forward(self, x: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        audio_feat = self.audio_model(audio_feat)
        x5 = self.fuse_conv(torch.cat([x5, audio_feat], dim=1))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


def reparameterize_model(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


if __name__ == "__main__":
    net = Model(6, "hubert").eval()
    img = torch.zeros(1, 6, 160, 160)
    audio = torch.zeros(1, 16, 32, 32)
    with torch.no_grad():
        out = net(img, audio)
    print(tuple(out.shape))
    print(sum(p.numel() for p in net.parameters()))
