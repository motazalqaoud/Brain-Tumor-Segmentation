"""
visualizer3d.py
---------------
Visualization utilities for 3D brain tumor segmentation results.

Displays:
- MRI input slices (axial, sagittal, coronal)
- Ground truth masks
- Model predictions
- Overlay comparison
- Metrics overlaid on images

Author: Motaz Alqaoud, PhD
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Tuple, Optional, List
import torch
import os
from pathlib import Path


class SegmentationVisualizer:
    """Visualize 3D segmentation results."""
    
    # Tumor class colors
    COLORS = {
        0: [0, 0, 0],           # background (black)
        1: [1, 0, 0],           # glioma (red)
        2: [0, 1, 0],           # meningioma (green)
        3: [0, 0, 1],           # pituitary (blue)
    }
    
    CLASS_NAMES = {
        0: 'Background',
        1: 'Glioma',
        2: 'Meningioma',
        3: 'Pituitary',
    }
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Args:
            output_dir: directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def visualize_2d_slice(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                          prediction: Optional[np.ndarray] = None,
                          title: str = '', save_path: Optional[str] = None,
                          metrics: Optional[dict] = None) -> plt.Figure:
        """
        Visualize a single 2D slice with ground truth and prediction.
        
        Args:
            image: (H, W) MRI slice
            mask: (H, W) ground truth mask (class indices)
            prediction: (H, W) predicted mask (class indices)
            title: plot title
            save_path: path to save figure
            metrics: dict of metrics to display (e.g., {'Dice': 0.85})
            
        Returns:
            matplotlib figure
        """
        n_cols = 1 + (mask is not None) + (prediction is not None)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        if n_cols == 1:
            axes = [axes]
        
        col = 0
        
        # Input image
        axes[col].imshow(image, cmap='gray')
        axes[col].set_title(f'MRI Input\n{title}', fontsize=12, fontweight='bold')
        axes[col].axis('off')
        col += 1
        
        # Ground truth
        if mask is not None:
            mask_rgb = self._mask_to_rgb(mask)
            axes[col].imshow(image, cmap='gray', alpha=0.6)
            axes[col].imshow(mask_rgb, alpha=0.5)
            axes[col].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
            axes[col].axis('off')
            col += 1
        
        # Prediction
        if prediction is not None:
            pred_rgb = self._mask_to_rgb(prediction)
            axes[col].imshow(image, cmap='gray', alpha=0.6)
            axes[col].imshow(pred_rgb, alpha=0.5)
            
            title_str = 'Prediction'
            if metrics:
                metric_str = ' | '.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
                title_str += f'\n{metric_str}'
            
            axes[col].set_title(title_str, fontsize=12, fontweight='bold')
            axes[col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_volume(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                           prediction: Optional[np.ndarray] = None,
                           title: str = '', save_path: Optional[str] = None,
                           num_slices: int = 6) -> plt.Figure:
        """
        Visualize multiple slices from 3D volume.
        
        Args:
            image: (D, H, W) 3D MRI volume
            mask: (D, H, W) ground truth
            prediction: (D, H, W) predicted mask
            title: figure title
            save_path: path to save
            num_slices: number of evenly spaced slices to show
            
        Returns:
            matplotlib figure
        """
        d, h, w = image.shape
        slice_indices = np.linspace(0, d-1, num_slices, dtype=int)
        
        n_cols = 1 + (mask is not None) + (prediction is not None)
        fig, axes = plt.subplots(num_slices, n_cols, figsize=(5*n_cols, 4*num_slices))
        
        for i, slice_idx in enumerate(slice_indices):
            col = 0
            
            # Input
            axes[i, col].imshow(image[slice_idx], cmap='gray')
            if i == 0:
                axes[i, col].set_title('MRI Input', fontsize=12, fontweight='bold')
            axes[i, col].set_ylabel(f'Slice {slice_idx}', fontsize=10)
            axes[i, col].axis('off')
            col += 1
            
            # Ground truth
            if mask is not None:
                mask_rgb = self._mask_to_rgb(mask[slice_idx])
                axes[i, col].imshow(image[slice_idx], cmap='gray', alpha=0.6)
                axes[i, col].imshow(mask_rgb, alpha=0.5)
                if i == 0:
                    axes[i, col].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[i, col].axis('off')
                col += 1
            
            # Prediction
            if prediction is not None:
                pred_rgb = self._mask_to_rgb(prediction[slice_idx])
                axes[i, col].imshow(image[slice_idx], cmap='gray', alpha=0.6)
                axes[i, col].imshow(pred_rgb, alpha=0.5)
                if i == 0:
                    axes[i, col].set_title('Prediction', fontsize=12, fontweight='bold')
                axes[i, col].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_segmentation_comparison(self, image: np.ndarray,
                                         mask: np.ndarray,
                                         prediction: np.ndarray,
                                         title: str = '',
                                         save_path: Optional[str] = None,
                                         metrics: Optional[dict] = None) -> plt.Figure:
        """
        Side-by-side comparison of ground truth vs prediction.
        
        Args:
            image: (H, W) MRI slice
            mask: (H, W) ground truth
            prediction: (H, W) predicted mask
            title: figure title
            save_path: path to save
            metrics: dict of metrics
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Input image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input MRI', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground truth overlay
        mask_rgb = self._mask_to_rgb(mask)
        axes[0, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[0, 1].imshow(mask_rgb, alpha=0.6)
        axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Prediction overlay
        pred_rgb = self._mask_to_rgb(prediction)
        axes[1, 0].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(pred_rgb, alpha=0.6)
        axes[1, 0].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Difference (prediction vs ground truth)
        diff = (prediction != mask).astype(np.uint8)
        diff_colored = np.zeros((*diff.shape, 3))
        diff_colored[diff > 0] = [1, 0, 0]  # Mismatch in red
        
        axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(diff_colored, alpha=0.6)
        axes[1, 1].set_title('Error Map (Red=Mismatch)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add metrics text
        if metrics:
            metric_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert class mask to RGB image.
        
        Args:
            mask: (H, W) with class indices
            
        Returns:
            (H, W, 3) RGB image
        """
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        
        for class_id, color in self.COLORS.items():
            rgb[mask == class_id] = color
        
        return rgb
    
    def plot_training_curves(self, train_losses: List[float],
                            val_losses: List[float],
                            train_dices: List[float],
                            val_dices: List[float],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            train_losses: list of training losses
            val_losses: list of validation losses
            train_dices: list of training Dice scores
            val_dices: list of validation Dice scores
            save_path: path to save
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curve
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Dice curve
        axes[1].plot(epochs, train_dices, 'b-', label='Train Dice', linewidth=2)
        axes[1].plot(epochs, val_dices, 'r-', label='Val Dice', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Dice Coefficient', fontsize=12)
        axes[1].set_title('Training & Validation Dice', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self, class_metrics: dict,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot per-class metrics (Dice, Hausdorff, etc.).
        
        Args:
            class_metrics: dict like {class_name: {metric_name: value}}
            save_path: path to save
            
        Returns:
            matplotlib figure
        """
        classes = list(class_metrics.keys())
        metrics = list(class_metrics[classes[0]].keys()) if classes else []
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [class_metrics[cls].get(metric, 0) for cls in classes]
            
            colors = ['red', 'green', 'blue', 'orange'][:len(classes)]
            axes[i].bar(classes, values, color=colors, alpha=0.7, edgecolor='black')
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].set_title(f'{metric} per Class', fontsize=12, fontweight='bold')
            axes[i].set_ylim([0, max(values) * 1.1])
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def save_sample_predictions(model, dataloader, visualizer: SegmentationVisualizer,
                           num_samples: int = 5, output_dir: str = 'predictions'):
    """
    Generate and save predictions for sample batch.
    
    Args:
        model: trained segmentation model
        dataloader: data loader
        visualizer: SegmentationVisualizer instance
        num_samples: number of samples to visualize
        output_dir: directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Predict
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Visualize each sample in batch
            for i in range(min(len(images), num_samples)):
                img_np = images[i, 0].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                pred_np = predictions[i].cpu().numpy()
                
                # Get middle slice if 3D
                if img_np.ndim == 3:
                    mid_slice = img_np.shape[0] // 2
                    img_np = img_np[mid_slice]
                    mask_np = mask_np[mid_slice]
                    pred_np = pred_np[mid_slice]
                
                # Normalize image
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                fig = visualizer.visualize_2d_slice(
                    image=img_np,
                    mask=mask_np,
                    prediction=pred_np,
                    title=f'Sample {batch_idx * len(images) + i}',
                    save_path=str(output_dir / f'sample_{batch_idx}_{i}.png')
                )
                plt.close(fig)
    
    print(f"Saved {num_samples} predictions to {output_dir}")
