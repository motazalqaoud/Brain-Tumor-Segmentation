"""
3D Attention U-Net for multi-label brain tumor segmentation -- operates
directly on the full (4, H, W, D) volumetric input (FLAIR, T1w, T1gd, T2w),
producing a (3, H, W, D) multi-label output (TC, WT, ET), one independent
sigmoid channel per region.

This keeps the "3D Attention U-Net" architecture already used elsewhere in
this portfolio (brain-tumor-segmentation), now correctly applied to real
volumetric input instead of 2D slices. MONAI's SegResNet (the architecture
that won BraTS 2018, "3D MRI Brain Tumor Segmentation Using Autoencoder
Regularization", Myronenko 2018) is a well-documented, strong alternative on
this exact dataset -- worth trying as a comparison/ablation, see README --
but AttentionUnet is what's used here to keep this consistent with the rest
of the portfolio.
"""

import torch
from monai.networks.nets import AttentionUnet


def build_model(in_channels: int = 4, out_channels: int = 3):
    return AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        dropout=0.1,
    )


def load_model(checkpoint_path: str = None, device: str = "cpu",
                in_channels: int = 4, out_channels: int = 3):
    """
    Returns (model, weights_loaded: bool). If checkpoint_path exists, loads
    the trained weights; otherwise returns a freshly (randomly) initialized
    network -- there is no ImageNet-style pretrained backbone for 3D medical
    volumes with 4 non-RGB input channels, so "no checkpoint" here genuinely
    means untrained, and callers (the Gradio app) must label that clearly.
    """
    import os
    model = build_model(in_channels=in_channels, out_channels=out_channels)
    weights_loaded = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        weights_loaded = True

    model.to(device)
    model.eval()
    return model, weights_loaded
