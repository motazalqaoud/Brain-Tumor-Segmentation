"""
losses.py
---------
Loss functions for medical image segmentation.

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    Standard cross-entropy fails badly on medical images because lesions
    are tiny compared to background (class imbalance can be 1000:1).
    Dice loss directly optimizes the overlap metric used clinically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Directly optimizes the Dice Similarity Coefficient (DSC),
    which is the standard evaluation metric in medical segmentation.

    DSC = 2 * |A ∩ B| / (|A| + |B|)
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model output (B, 1, H, W) — before sigmoid
            targets: binary ground truth (B, 1, H, W), values in {0, 1}

        Returns:
            Dice loss scalar (1 - DSC)
        """
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    
    BCE handles pixel-level accuracy.
    Dice handles region-level overlap.
    Together they train faster and more stably than either alone.

    Loss = alpha * BCE + (1 - alpha) * Dice
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0,
                 pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            alpha: weight for BCE component (0.5 = equal weighting)
            smooth: Dice smoothing factor
            pos_weight: optional weight for positive class (handles class imbalance)
        """
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy negatives, focuses on hard examples.
    
    Useful when the lesion is very small (e.g. early-stage tumor < 5mm).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: weighting factor for positive class
            gamma: focusing parameter (0 = standard BCE, 2 = standard focal)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return focal_loss.mean()


# ─── Metrics (not losses, but used together) ─────────────────────────────────

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Compute Dice Similarity Coefficient for evaluation (not training).

    Args:
        pred: sigmoid probabilities or logits (B, 1, H, W)
        target: binary ground truth (B, 1, H, W)
        threshold: binarization threshold
        smooth: smoothing factor

    Returns:
        DSC value in [0, 1]
    """
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)

    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dsc = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dsc.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Intersection over Union (Jaccard Index).

    Args:
        pred: probabilities (B, 1, H, W)
        target: binary ground truth
        threshold: binarization threshold

    Returns:
        IoU value in [0, 1]
    """
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)

    pred_bin = (pred > threshold).float().view(-1)
    target_flat = target.float().view(-1)

    intersection = (pred_bin * target_flat).sum()
    union = pred_bin.sum() + target_flat.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()
