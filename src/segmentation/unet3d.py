"""
unet3d.py
---------
3D Attention U-Net for brain tumor segmentation.

Architecture: 3D convolutional encoder-decoder with squeeze-and-excitation (SE)
and spatial attention gates for focusing on tumor regions.

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    3D models leverage volumetric context (adjacent slices) for better tumor
    segmentation. Attention mechanisms help the model focus on small lesions
    that might otherwise be ignored in noisy brain images.

Target performance: Dice >0.90 on multi-class brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel attention.
    
    Recalibrates channel-wise feature responses by learning to weight
    channels based on their importance.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc(se)
        se = self.sigmoid(se)
        return x * se


class SpatialAttention(nn.Module):
    """Spatial attention gate.
    
    Creates a spatial mask that emphasizes tumor-relevant regions.
    Uses both channel and spatial information.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        mask = self.conv(x)
        return x * mask


class AttentionGate(nn.Module):
    """Attention gate for skip connections (decoder).
    
    Combines channel and spatial attention at decoder levels to focus
    on relevant features from encoder skip connections.
    """
    
    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        Args:
            f_g: channels in gating signal (from deeper layer)
            f_l: channels in skip connection (from encoder)
            f_int: channels in intermediate representation
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm3d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm3d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Conv3DBlock(nn.Module):
    """Residual 3D convolution block with attention."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None
        
        # Squeeze-and-excitation attention
        self.channel_attention = ChannelAttention(out_channels) if use_attention else None
        
        # Residual connection if channels match
        self.skip = (in_channels == out_channels)
        if not self.skip and in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply channel attention
        if self.channel_attention:
            out = self.channel_attention(out)
        
        # Add skip connection
        if self.skip:
            out += identity
        elif hasattr(self, 'skip_conv'):
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)
            out += identity
        
        out = self.relu(out)
        return out


class Down3D(nn.Module):
    """Downsampling: MaxPool3D → Conv3DBlock."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        # Pool only H and W; preserve D so pseudo-3D inputs (small D) don't collapse.
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = Conv3DBlock(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up3D(nn.Module):
    """Upsampling → concatenate skip → Conv3DBlock with attention gate."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_attention_gate: bool = True):
        super().__init__()
        # Upsample only H and W to match the anisotropic downsampling in Down3D.
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Attention gate for skip connection
        if use_attention_gate:
            self.attention = AttentionGate(
                f_g=in_channels // 2,
                f_l=skip_channels,
                f_int=skip_channels // 2
            )
        else:
            self.attention = None
        
        self.conv = Conv3DBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            dropout=0.0
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatches
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2,
                      diff_d // 2, diff_d - diff_d // 2])
        
        # Apply attention gate to skip connection
        if self.attention:
            skip = self.attention(x, skip)
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for multi-class brain tumor segmentation.
    
    Features:
    - 3D volumetric processing (preserves spatial context)
    - Residual conv blocks with channel attention
    - Spatial attention gates in decoder
    - Multi-class output: background + 7 WHO tumor categories (8 classes total)

    Args:
        in_channels: input channels (1 for single MRI modality)
        num_classes: number of output classes (8 = background + 7 WHO tumor categories)
        base_filters: initial number of filters (doubled each level)
        depth: number of encoder levels (default 4 = 8x8x8 bottleneck)
        dropout: dropout rate in encoder

    Example:
        >>> model = AttentionUNet3D(in_channels=1, num_classes=8)
        >>> x = torch.randn(2, 1, 4, 64, 64)
        >>> out = model(x)
        >>> print(out.shape)  # (2, 8, 4, 64, 64)
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 base_filters: int = 32,
                 depth: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.depth = depth
        
        # Encoder (down path)
        self.initial_conv = Conv3DBlock(in_channels, base_filters, dropout=0.0)
        
        self.down_layers = nn.ModuleList()
        self.down_channels = [base_filters]
        
        ch = base_filters
        for i in range(depth):
            drop = dropout if i == depth - 1 else 0.0
            ch_out = ch * 2
            self.down_layers.append(Down3D(ch, ch_out, dropout=drop))
            self.down_channels.append(ch_out)
            ch = ch_out
        
        # Bottleneck
        self.bottleneck = Conv3DBlock(ch, ch, dropout=dropout)
        
        # Decoder (up path)
        self.up_layers = nn.ModuleList()
        for i in range(depth):
            ch_in = ch
            ch_skip = self.down_channels[-(i + 2)]
            ch_out = ch_skip
            self.up_layers.append(
                Up3D(ch_in, ch_skip, ch_out, use_attention_gate=True)
            )
            ch = ch_out
        
        # Output layer
        self.final_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input volume (B, C, D, H, W) where D/H/W are spatial dimensions
            
        Returns:
            logits: (B, num_classes, D, H, W)
        """
        # Encoder path
        skips = []
        x = self.initial_conv(x)
        skips.append(x)
        
        for down in self.down_layers:
            x = down(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, up in enumerate(self.up_layers):
            skip = skips[-(i + 2)]
            x = up(x, skip)
        
        # Output
        logits = self.final_conv(x)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference and return class predictions.
        
        Args:
            x: input volume (B, C, D, H, W)
            
        Returns:
            predictions: (B, D, H, W) with class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = AttentionUNet3D(in_channels=1, num_classes=8, base_filters=32, depth=4)
    print(f"Model parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 1, 128, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    pred = model.predict(x)
    print(f"Prediction shape: {pred.shape}")
    print(f"Class indices: {pred.unique()}")
