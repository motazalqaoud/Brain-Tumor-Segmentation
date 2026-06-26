"""
test_model.py
-------------
Quick test to verify model, dataset, and visualization setup.

Usage:
    python scripts/test_model.py --data-root data/raw/Images_
"""

import argparse
import sys
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation.unet3d import AttentionUNet3D
from preprocessing.brain_tumor_loader import BrainTumorDataset
from visualization.visualizer3d import SegmentationVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                       default=str(Path(__file__).parent.parent / 'data' / 'raw' / 'Images_'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=3)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    logger.info(f"Testing on device: {device}")
    
    # Test 1: Load dataset
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Dataset Loading")
    logger.info("="*60)
    
    try:
        dataset = BrainTumorDataset(
            root_dir=args.data_root,
            modality='T1c',
            split='train',
            mode='slice'
        )
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Get samples
        samples = [dataset[i] for i in range(min(args.num_samples, len(dataset)))]
        for i, sample in enumerate(samples):
            logger.info(f"  Sample {i}: Image {sample['image'].shape}, "
                       f"Mask {sample['mask'].shape}, "
                       f"Tumor: {sample['tumor_type']}")
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        return
    
    # Test 2: Create model
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Model Creation")
    logger.info("="*60)
    
    try:
        model = AttentionUNet3D(
            in_channels=1,
            num_classes=4,
            base_filters=32,
            depth=3
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model created: {num_params:,} parameters")
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return
    
    # Test 3: Forward pass
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Forward Pass")
    logger.info("="*60)
    
    try:
        model.eval()
        
        # Create dummy 3D batch (B, C, D, H, W)
        dummy_input = torch.randn(1, 1, 64, 128, 128).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Input shape: {tuple(dummy_input.shape)}")
        logger.info(f"  Output shape: {tuple(output.shape)}")
        logger.info(f"  Output classes: {output.shape[1]}")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return
    
    # Test 4: Inference on real data
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Inference on Real Data (2D to 3D)")
    logger.info("="*60)
    
    try:
        from torch.utils.data import DataLoader
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        images = batch['image'].to(device)  # (B, 1, H, W)
        masks = batch['mask'].to(device)    # (B, H, W)
        
        logger.info(f"  Raw batch images shape: {tuple(images.shape)}")
        logger.info(f"  Raw batch masks shape: {tuple(masks.shape)}")
        
        # Convert 2D to 3D by stacking slices (create pseudo-volume)
        # Stack same slice 8 times to create (B, 1, 8, H, W) volume
        images_3d = images.unsqueeze(2).repeat(1, 1, 8, 1, 1)  # (B, 1, 8, H, W)
        masks_3d = masks.unsqueeze(1).repeat(1, 8, 1, 1)  # (B, 8, H, W)
        
        logger.info(f"  3D images shape: {tuple(images_3d.shape)}")
        logger.info(f"  3D masks shape: {tuple(masks_3d.shape)}")
        
        with torch.no_grad():
            logits = model(images_3d)
            predictions = torch.argmax(logits, dim=1)
        
        logger.info(f"✓ Inference successful")
        logger.info(f"  Logits shape: {tuple(logits.shape)}")
        logger.info(f"  Predictions shape: {tuple(predictions.shape)}")
        logger.info(f"  Unique classes in predictions: {predictions.unique().tolist()}")
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        return
    
    # Test 5: Visualization
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Visualization")
    logger.info("="*60)
    
    try:
        viz_dir = Path('test_visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        visualizer = SegmentationVisualizer(output_dir=str(viz_dir))
        
        # Use 2D slices from batch (extract middle of 3D volume)
        img_np = images_3d[0, 0, 4].cpu().numpy()  # Middle slice
        mask_np = masks_3d[0, 4].cpu().numpy()
        pred_np = predictions[0, 4].cpu().numpy()
        
        # Normalize image
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Save visualization
        fig = visualizer.visualize_segmentation_comparison(
            image=img_np,
            mask=mask_np,
            prediction=pred_np,
            title='Test Visualization (2D Slice)',
            save_path=str(viz_dir / 'test_visualization.png')
        )
        plt.close(fig)
        
        logger.info(f"✓ Visualization saved to {viz_dir / 'test_visualization.png'}")
    except Exception as e:
        logger.error(f"✗ Visualization failed: {e}")
        return
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("="*60)
    logger.info("\nYou can now start training with:")
    logger.info(f"  python scripts/train3d.py --epochs 50 --batch 4 --lr 1e-3")
    logger.info("="*60)


if __name__ == '__main__':
    main()
