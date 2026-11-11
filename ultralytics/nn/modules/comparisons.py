# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Reference operators for controlled MCGI-YOLO comparisons."""
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
