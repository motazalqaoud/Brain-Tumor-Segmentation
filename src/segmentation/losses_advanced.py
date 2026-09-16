"""
losses_advanced.py
------------------
Advanced loss functions for multi-class brain tumor segmentation.

Includes:
- Weighted Dice Loss (handles class imbalance)
- Focal Loss (handles hard examples, small tumors)
- Boundary Loss (preserves tumor boundaries)
- Hybrid Loss (combines all three)

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    Brain tumors are small compared to background (~5-10% of volume).
    Standard cross-entropy fails. We need:
    1. Dice to optimize overlap (clinical metric)
    2. Focal to focus on hard negatives
    3. Boundary to preserve surgical margins
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class WeightedDiceLoss(nn.Module):
    """
    Multi-class Weighted Dice Loss.
    
    Dice_class_c = 2 * |A_c ∩ B_c| / (|A_c| + |B_c|)
    Loss = sum(w_c * (1 - Dice_c)) for all classes c
    
    Weights compensate for class imbalance (background huge, glioma rare).
    """
    
    def __init__(self, num_classes: int = 4, weights: Optional[List[float]] = None,
                 smooth: float = 1.0, ignore_index: int = -1):
        """
        Args:
            num_classes: number of output classes
            weights: per-class weights. If None, computed from frequencies.
                     default: [0.01, 0.4, 0.3, 0.29] for tumor types
            smooth: smoothing factor to avoid division by zero
            ignore_index: class to ignore in loss (-1 = no ignore)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        
        if weights is None:
            # Inverse frequency weighting for common brain tumor classes
            if num_classes == 4:
                weights = [0.01, 0.4, 0.3, 0.29]  # bg, glioma, meningioma, pituitary
            else:
                weights = [1.0 / num_classes] * num_classes
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth class indices
            
        Returns:
            weighted dice loss
        """
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        
        dice_per_class = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            
            # One-hot encode this class
            target_c = (targets == c).float()  # (B, D, H, W)
            prob_c = probs[:, c, :, :, :]     # (B, D, H, W)
            
            # Flatten spatial dimensions
            prob_flat = prob_c.view(prob_c.shape[0], -1)
            target_flat = target_c.view(target_c.shape[0], -1)
            
            # Dice coefficient
            intersection = (prob_flat * target_flat).sum(dim=1)
            dice_c = (2.0 * intersection + self.smooth) / (
                prob_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
            )
            dice_per_class.append(dice_c)
        
        dice_per_class = torch.stack(dice_per_class, dim=1)  # (B, C-1)
        weighted_dice = torch.mean(1.0 - dice_per_class, dim=0)
        
        return weighted_dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Useful when tumors are tiny (class imbalance 1:1000+).
    Focuses training on hard negatives.
    """
    
    def __init__(self, num_classes: int = 4, alpha: Optional[List[float]] = None,
                 gamma: float = 2.0):
        """
        Args:
            num_classes: number of classes
            alpha: per-class weighting (usually inverse frequency)
            gamma: focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        
        if alpha is None:
            alpha = [1.0] * num_classes
        
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth class indices
            
        Returns:
            focal loss
        """
        # Standard cross-entropy
        ce = F.cross_entropy(logits, targets, reduction='none')  # (B, D, H, W)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, D, H, W)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply per-class alpha weights
        alpha_t = self.alpha[targets]  # (B, D, H, W)
        
        focal = alpha_t * focal_weight * ce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: Penalizes boundary voxels more heavily.
    
    Useful for ensuring tumor margins are precise (critical for surgery).
    Uses Hausdorff distance on boundaries.
    """
    
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, D, H, W)
            
        Returns:
            boundary loss
        """
        probs = F.softmax(logits, dim=1)
        
        boundary_loss = 0.0
        for c in range(1, self.num_classes):  # Skip background
            # One-hot
            target_c = (targets == c).float()
            prob_c = probs[:, c, :, :, :]
            
            # Compute boundary by morphological gradient
            # (dilation - erosion) gives boundary
            kernel = torch.ones(1, 1, 3, 3, 3, device=target_c.device) / 27
            
            # Dilated target
            target_expanded = F.conv3d(
                target_c.unsqueeze(1),
                kernel,
                padding=1
            ).clamp(0, 1)
            
            # Eroded target (approximated by 1 - dilated(1-target))
            target_eroded = 1 - F.conv3d(
                (1 - target_c).unsqueeze(1),
                kernel,
                padding=1
            ).clamp(0, 1)
            
            # Boundary
            boundary = (target_expanded - target_eroded).squeeze(1)
            
            # Penalty on boundary voxels
            boundary_loss += (boundary * (1 - prob_c)).mean()
        
        return boundary_loss / max(1, self.num_classes - 1)


class HybridLoss(nn.Module):
    """
    Hybrid Loss: Combines Weighted Dice + Focal + Boundary
    
    Loss = α * Dice + β * Focal + γ * Boundary
    
    This addresses:
    - Dice: optimizes the clinical metric (overlap)
    - Focal: handles hard examples and tiny tumors
    - Boundary: preserves surgical margins
    """
    
    def __init__(self, num_classes: int = 4,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 dice_weights: Optional[List[float]] = None,
                 focal_alpha: Optional[List[float]] = None,
                 focal_gamma: float = 2.0):
        """
        Args:
            num_classes: number of classes
            alpha: weight for Dice component
            beta: weight for Focal component
            gamma: weight for Boundary component
            dice_weights: per-class weights for Dice
            focal_alpha: per-class weights for Focal
            focal_gamma: focusing parameter for Focal
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.dice_loss = WeightedDiceLoss(num_classes, weights=dice_weights)
        self.focal_loss = FocalLoss(num_classes, alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss(num_classes)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, D, H, W)
            
        Returns:
            hybrid loss value
        """
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        
        total = self.alpha * dice + self.beta * focal + self.gamma * boundary
        
        return total


# ─── Metrics ──────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor,
               class_idx: int, smooth: float = 1.0) -> float:
    """
    Compute Dice score for a single class.
    
    Args:
        pred: (B, C, D, H, W) or (B, D, H, W) predictions
        target: (B, D, H, W) ground truth
        class_idx: which class to compute Dice for
        
    Returns:
        Dice value [0, 1]
    """
    if pred.dim() == 5:
        pred = torch.argmax(pred, dim=1)
    
    pred_c = (pred == class_idx).float()
    target_c = (target == class_idx).float()
    
    intersection = (pred_c * target_c).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_c.sum() + target_c.sum() + smooth
    )
    return dice.item()


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor,
                      class_idx: int) -> float:
    """
    Compute Hausdorff distance for a single class.
    
    Hausdorff = max(min(d(p, target)) for p in pred)
    
    Measures worst-case boundary error.
    """
    if pred.dim() == 5:
        pred = torch.argmax(pred, dim=1)
    
    pred_mask = (pred == class_idx).cpu().numpy()
    target_mask = (target == class_idx).cpu().numpy()
    
    # Simple version: compute max distance
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return 0.0
    
    pred_indices = np.where(pred_mask)
    target_indices = np.where(target_mask)
    
    if len(pred_indices[0]) == 0 or len(target_indices[0]) == 0:
        return np.inf
    
    # Min distance from each pred point to target
    max_dist = 0
    for i in range(len(pred_indices[0])):
        point = np.array([pred_indices[j][i] for j in range(3)])
        target_points = np.array([target_indices[j] for j in range(3)])
        min_dist = np.linalg.norm(target_points - point[:, None], axis=0).min()
        max_dist = max(max_dist, min_dist)
    
    return float(max_dist)


def volume_correlation(pred_volume: float, target_volume: float) -> float:
    """
    Compute correlation between predicted and target tumor volumes.
    
    Important for prognosis and surgical planning.
    """
    if target_volume < 1.0:
        return 1.0 if pred_volume < 1.0 else 0.0
    
    return min(pred_volume, target_volume) / max(pred_volume, target_volume)
