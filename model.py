"""Three-Stage Cascaded RCAN with global residual learning.

SEMICON India Hackathon 2026 — Track 1 (KLA), PS01.

4 621 017 parameters. Three cascaded stages of residual groups with channel
attention, a fusion of stage-1 and stage-2 features, PixelShuffle x2, and a
tail that predicts a RESIDUAL over a bilinear upsample of the input — so the
network learns the correction, not the image.

Input is 3-channel: the raw noisy array, a bilateral-filtered copy, and a
log-domain Fourier-filtered copy (see inference.preprocess).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-excite over channels: pool to 1x1, bottleneck, gate."""

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True),
            ChannelAttention(num_channels, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, num_channels, num_blocks, reduction=16):
        super().__init__()
        mods = [RCAB(num_channels, reduction) for _ in range(num_blocks)]
        mods.append(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True))
        self.body = nn.Sequential(*mods)

    def forward(self, x):
        return x + self.body(x)


class Stage(nn.Module):
    def __init__(self, num_channels, num_groups, num_blocks):
        super().__init__()
        mods = [ResidualGroup(num_channels, num_blocks) for _ in range(num_groups)]
        mods.append(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=True))
        self.body = nn.Sequential(*mods)

    def forward(self, x):
        return x + self.body(x)


class CascadedRCAN(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_groups=3, num_blocks=6):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.stage1 = Stage(num_channels, num_groups, num_blocks)
        self.stage2 = Stage(num_channels, num_groups, num_blocks)
        self.fusion = nn.Conv2d(num_channels * 2, num_channels, 1)
        self.stage3 = Stage(num_channels, num_groups, num_blocks)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.tail = nn.Conv2d(num_channels, 1, 3, padding=1)

    def forward(self, x):
        # channel 0 is the raw noisy input; the network predicts a correction to
        # its bilinear upsample, so failure degrades toward interpolation rather
        # than toward invented structure
        base = F.interpolate(x[:, 0:1], scale_factor=2, mode='bilinear',
                             align_corners=False)
        h = self.head(x)
        s1 = self.stage1(h)
        s2 = self.stage2(s1)
        fused = self.fusion(torch.cat([s1, s2], dim=1))
        s3 = self.stage3(fused)
        return base + self.tail(self.upsample(s3))


def build_model(in_channels=3, num_channels=64, num_groups=3, num_blocks=6):
    return CascadedRCAN(in_channels, num_channels, num_groups, num_blocks)


if __name__ == '__main__':
    m = build_model()
    n = sum(p.numel() for p in m.parameters())
    print(f'CascadedRCAN: {n:,} parameters')
    y = m(torch.randn(1, 3, 128, 128))
    print('128x128 ->', tuple(y.shape[-2:]))
