# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Reference operators for controlled MCGI-YOLO comparisons; author MIT notices: comparisons.LICENSE."""
import torch
from torch import nn
from torch.nn import functional as F
from .block import C2f


class CoordinateAttention(nn.Module):
    """Coordinate Attention (CVPR 2021), reduction=32 as in houqb/CoordAttention."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.height = nn.Conv2d(hidden, channels, 1)
        self.width = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        rows = x.mean(3, keepdim=True)
        columns = x.mean(2, keepdim=True).transpose(2, 3)
        descriptor = self.act(self.bn(self.reduce(torch.cat((rows, columns), 2))))
        rows, columns = descriptor.split((h, w), 2)
        return x * self.height(rows).sigmoid() * self.width(columns.transpose(2, 3)).sigmoid()


class C2fCoordinateAttention(C2f):
    """Apply unmodified CA at MCE's site: final hidden bottleneck output before C2f concatenation."""

    def __init__(self, c1, c2, n=1, shortcut=False, reduction=32):
        super().__init__(c1, c2, n, shortcut)
        self.ca = CoordinateAttention(self.c, reduction)

    def _aggregate(self, parts):
        parts.extend(layer(parts[-1]) for layer in self.m)
        parts[-1] = self.ca(parts[-1])
        return self.cv2(torch.cat(parts, 1))

    def forward(self, x):
        return self._aggregate(list(self.cv1(x).chunk(2, 1)))

    def forward_split(self, x):
        return self._aggregate(list(self.cv1(x).split((self.c, self.c), 1)))


class EfficientChannelAttention(nn.Module):
    """ECA with the author implementation's default 3-tap channel convolution."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if channels < 1 or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Require positive channels and an odd positive ECA kernel size.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        descriptor = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(descriptor).sigmoid().transpose(1, 2).unsqueeze(-1)
        return x * weights


class CBAMReference(nn.Module):
    """Author CBAM: shared MLP on avg/max descriptors, then max/mean spatial pooling and Conv-BN."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if reduction < 1 or channels < reduction:
            raise ValueError("CBAM requires channels >= reduction >= 1.")
        hidden = channels // reduction
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def forward(self, x):
        average = self.mlp(x.mean((2, 3)))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1).flatten(1))
        x = x * (average + maximum).sigmoid().unsqueeze(-1).unsqueeze(-1)
        descriptor = torch.cat((x.max(1, keepdim=True).values, x.mean(1, keepdim=True)), 1)
        return x * self.spatial_bn(self.spatial(descriptor)).sigmoid()


class _C2fFinalBranchAttention(C2f):
    """Retain C2f weight paths and apply a reference operator at MCE's final hidden branch."""

    def _aggregate(self, parts):
        parts.extend(layer(parts[-1]) for layer in self.m)
        parts[-1] = self.attention(parts[-1])
        return self.cv2(torch.cat(parts, 1))

    def forward(self, x):
        return self._aggregate(list(self.cv1(x).chunk(2, 1)))

    def forward_split(self, x):
        return self._aggregate(list(self.cv1(x).split((self.c, self.c), 1)))


class C2fECA(_C2fFinalBranchAttention):
    """Replace MCE with ECA, leaving all original C2f bottlenecks intact."""

    def __init__(self, c1, c2, n=1, shortcut=False, kernel_size=3):
        super().__init__(c1, c2, n, shortcut)
        self.attention = EfficientChannelAttention(self.c, kernel_size)


class C2fCBAM(_C2fFinalBranchAttention):
    """Replace MCE with the author CBAM, rather than the simplified conv.CBAM variant."""

    def __init__(self, c1, c2, n=1, shortcut=False, reduction=16):
        super().__init__(c1, c2, n, shortcut)
        self.attention = CBAMReference(self.c, reduction)


class DySampleUpsample(nn.Module):
    """DySample's default LP operator (scale=2, groups=4, dyscope=False).

    Adapted from https://github.com/tiny-smart/dysample/blob/main/dysample.py.
    ECA, CBAM, and DySample MIT notices are retained in comparisons.LICENSE.
    """

    def __init__(self, c1, scale_factor=2, groups=4, dyscope=False):
        super().__init__()
        if scale_factor != 2 or groups < 1 or c1 < groups or c1 % groups:
            raise ValueError("Require 2x upsampling and positive groups dividing the input channels.")
        self.scale_factor = scale_factor
        self.groups = groups
        self.dyscope = dyscope
        out_channels = 2 * groups * scale_factor**2
        self.offset = nn.Conv2d(c1, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        if dyscope:
            self.scope = nn.Conv2d(c1, out_channels, 1, bias=False)
            nn.init.zeros_(self.scope.weight)
        positions = torch.arange((-scale_factor + 1) / 2, (scale_factor - 1) / 2 + 1) / scale_factor
        origin = torch.stack(torch.meshgrid(positions, positions, indexing="ij")).transpose(1, 2)
        self.register_buffer("init_pos", origin.repeat(1, groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x):
        offset = self.offset(x)
        if self.dyscope:
            offset = offset * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = offset * 0.25 + self.init_pos
        b, _, h, w = x.shape
        offset = offset.view(b, 2, -1, h, w)
        width = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        height = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coordinates = torch.stack(torch.meshgrid(width, height, indexing="ij")).transpose(1, 2)
        coordinates = coordinates.unsqueeze(0).unsqueeze(2)
        normalizer = x.new_tensor((w, h)).view(1, 2, 1, 1, 1)
        coordinates = 2 * (coordinates + offset) / normalizer - 1
        coordinates = F.pixel_shuffle(coordinates.reshape(b, -1, h, w), self.scale_factor)
        coordinates = coordinates.view(b, 2, -1, 2 * h, 2 * w).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w), coordinates,
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        return sampled.view(b, -1, 2 * h, 2 * w)


class CARAFEUpsample(nn.Module):
    """Pure-PyTorch CARAFEPack equivalent (group=1), matching MMCV's default kernels and initialization.

    Source: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/carafe.html
    This is distinct from the repository's historical CAFR/ContentAwareUpsample variant.
    """

    def __init__(self, c1, scale_factor=2, up_kernel=5, compressed_channels=64, encoder_kernel=3):
        super().__init__()
        if scale_factor != 2 or up_kernel < 1 or up_kernel % 2 != 1:
            raise ValueError("Require 2x upsampling and a positive odd reassembly kernel.")
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.channel_compressor = nn.Conv2d(c1, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            compressed_channels, scale_factor**2 * up_kernel**2, encoder_kernel,
            padding=encoder_kernel // 2,
        )
        nn.init.xavier_uniform_(self.channel_compressor.weight)
        nn.init.zeros_(self.channel_compressor.bias)
        nn.init.normal_(self.content_encoder.weight, std=0.001)
        nn.init.zeros_(self.content_encoder.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        masks = F.pixel_shuffle(self.content_encoder(self.channel_compressor(x)), self.scale_factor).softmax(1)
        patches = F.unfold(x, self.up_kernel, padding=self.up_kernel // 2)
        patches = F.interpolate(patches.reshape(b, c * self.up_kernel**2, h, w), scale_factor=2, mode="nearest")
        patches = patches.reshape(b, c, self.up_kernel**2, 2 * h, 2 * w)
        return (patches * masks.unsqueeze(1)).sum(2)
