"""
unet.py
-------
U-Net architecture for medical image segmentation.

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.

Clinical note:
    U-Net is the backbone of most FDA-cleared AI segmentation tools.
    This implementation supports both 2D (slice-based) and configurable 
    depth for different organ/lesion sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DoubleConv(nn.Module):
    """Two consecutive Conv → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int = None, dropout: float = 0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(p=dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling: MaxPool → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling → concatenate skip connection → DoubleConv.

    Args:
        in_channels:   channels of the incoming (deeper) feature map x1
        skip_channels: channels of the skip-connection feature map x2
        out_channels:  channels produced by this block
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            # Upsample keeps channel count, so conv sees in_channels + skip_channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels,
                                   dropout=dropout)
        else:
            # Transposed conv halves the channels of x1
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                          kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels,
                                   dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed (handles odd input sizes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map to number of classes."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for 2D medical image segmentation.

    Args:
        in_channels:  number of input channels (1 for grayscale MRI)
        n_classes:    number of output classes (1 for binary tumor segmentation)
        base_filters: number of filters in first layer (doubles each down-step)
        depth:        number of encoder/decoder levels (default 4)
        bilinear:     use bilinear upsampling (True) or transposed conv (False)
        dropout:      dropout rate in encoder bottleneck

    Example:
        >>> model = UNet(in_channels=1, n_classes=1)
        >>> x = torch.randn(2, 1, 256, 256)
        >>> out = model(x)
        >>> print(out.shape)  # (2, 1, 256, 256)
    """

    def __init__(self,
                 in_channels: int = 1,
                 n_classes: int = 1,
                 base_filters: int = 64,
                 depth: int = 4,
                 bilinear: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Build encoder
        self.inc = DoubleConv(in_channels, base_filters)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        f = base_filters
        encoder_channels = [f]

        for i in range(depth):
            f_out = f * 2
            drop = dropout if i == depth - 1 else 0.0
            self.downs.append(Down(f, f_out, dropout=drop))
            encoder_channels.append(f_out)
            f = f_out

        # Build decoder — track the actual channel count flowing through.
        prev_ch = encoder_channels[-1]            # bottleneck output channels
        for i in range(depth):
            skip_ch = encoder_channels[-(i + 2)]  # matching encoder skip
            out_ch = skip_ch
            self.ups.append(Up(prev_ch, skip_ch, out_ch, bilinear, dropout=0.0))
            prev_ch = out_ch

        self.outc = OutConv(prev_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        skip_connections = [self.inc(x)]
        for down in self.downs:
            skip_connections.append(down(skip_connections[-1]))

        # Decoder path
        x = skip_connections[-1]
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[-(i+2)])

        logits = self.outc(x)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Run inference and return binary mask.

        Args:
            x: input tensor (B, C, H, W)
            threshold: sigmoid threshold for binary segmentation

        Returns:
            Binary mask tensor (B, 1, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.n_classes == 1:
                probs = torch.sigmoid(logits)
                return (probs > threshold).float()
            else:
                return torch.argmax(logits, dim=1, keepdim=True).float()

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(config: dict = None) -> UNet:
    """
    Factory function for building U-Net from a config dict.

    Default config is suitable for breast MRI tumor segmentation.

    Args:
        config: optional dict with keys matching UNet __init__ args

    Returns:
        UNet model
    """
    defaults = {
        'in_channels': 1,
        'n_classes': 1,
        'base_filters': 32,
        'depth': 4,
        'bilinear': True,
        'dropout': 0.1,
    }
    if config:
        defaults.update(config)

    model = UNet(**defaults)
    print(f"U-Net built: {model.count_parameters():,} trainable parameters")
    return model
