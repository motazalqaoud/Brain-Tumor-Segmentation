"""
train3d.py
----------
Training script for 3D Attention U-Net on brain tumor segmentation.

Features:
- 3D volumetric training
- Hybrid loss (Dice + Focal + Boundary)
- Learning rate scheduling
- Checkpoint saving
- Real-time visualization
- Multi-class brain tumor classification

Usage:
    python scripts/train3d.py --epochs 100 --batch 4 --lr 1e-3
    python scripts/train3d.py --epochs 50 --batch 8 --device cuda

Author: Motaz Alqaoud, PhD
"""

import argparse
import sys
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation.unet3d import AttentionUNet3D
from segmentation.losses_advanced import HybridLoss, dice_score
from preprocessing.brain_tumor_loader import BrainTumorDataset
from visualization.visualizer3d import SegmentationVisualizer

# Setup logging — writes to both terminal and training.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='a'),
    ],
    force=True,  # clears any pre-existing handlers to avoid duplicate lines
)
logger = logging.getLogger(__name__)

# Pseudo-3D depth: number of frames stacked from each 2D slice.
# Increase on GPU with more VRAM (e.g. 8 or 16); 4 is safe on CPU / 8GB RAM.
D_FRAMES = 2


def resolve_data_root(root_dir: str) -> str:
    """Normalize the dataset root path and support nested Images_ directories."""
    root = Path(root_dir)
    if root.exists() and root.is_dir():
        nested = root / 'Images_'
        if nested.exists() and nested.is_dir():
            logger.info(f"Detected nested Images_ directory, using: {nested}")
            return str(nested)
    return str(root)


def create_dataloaders(data_root: str, batch_size: int, num_workers: int = 0):
    """Create train / val / test dataloaders."""
    logger.info("Loading datasets...")

    train_ds = BrainTumorDataset(root_dir=data_root, modality='all', split='train', mode='slice')
    val_ds   = BrainTumorDataset(root_dir=data_root, modality='all', split='val',   mode='slice')
    test_ds  = BrainTumorDataset(root_dir=data_root, modality='all', split='test',  mode='slice')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    logger.info(f"Train set: {len(train_ds)} samples")
    logger.info(f"Val set:   {len(val_ds)} samples")
    logger.info(f"Test set:  {len(test_ds)} samples")

    return train_loader, val_loader, test_loader


TUMOR_CLASSES = [1, 2, 3, 4, 5, 6, 7]  # all tumor classes — excludes background (0)
CLASS_NAMES   = {
    1: 'Glioma',
    2: 'Meningioma',
    3: 'Nerve Sheath',
    4: 'Embryonic',
    5: 'Mixed Neuronal',
    6: 'Mesenchymal',
    7: 'Germ Cell',
}


def _mean_tumor_dice(class_dice_means: dict) -> float:
    """Mean Dice across tumor classes (1, 2, 3) — excludes background."""
    return float(np.mean([class_dice_means[c] for c in TUMOR_CLASSES]))


def train_epoch(model, loader, criterion, optimizer, device, image_size=64):
    """Train for one epoch. Returns (avg_loss, class_dice_means, mean_tumor_dice)."""
    model.train()
    total_loss = 0.0
    class_totals = {c: 0.0 for c in TUMOR_CLASSES}

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)

        images = F.interpolate(images, size=(image_size, image_size),
                               mode='bilinear', align_corners=False)
        masks  = F.interpolate(masks.unsqueeze(1).float(), size=(image_size, image_size),
                               mode='nearest').squeeze(1).long()

        images = images.unsqueeze(2).repeat(1, 1, D_FRAMES, 1, 1)
        masks  = masks.unsqueeze(1).repeat(1, D_FRAMES, 1, 1)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            for c in TUMOR_CLASSES:
                class_totals[c] += dice_score(preds, masks, class_idx=c)

        if (batch_idx + 1) % 10 == 0:
            n = batch_idx + 1
            avg_tumor = np.mean([class_totals[c] / n for c in TUMOR_CLASSES])
            logger.info(f"  Batch [{n}/{len(loader)}] Loss: {total_loss/n:.4f} | "
                        f"Tumor Dice: {avg_tumor:.4f}")

    n = len(loader)
    class_dice_means = {c: class_totals[c] / n for c in TUMOR_CLASSES}
    return total_loss / n, class_dice_means, _mean_tumor_dice(class_dice_means)


def evaluate(model, loader, criterion, device, num_classes=8, image_size=64, split='val'):
    """Run evaluation on any split. Returns (avg_loss, class_dice_means, mean_tumor_dice)."""
    model.eval()
    total_loss = 0.0
    class_dices = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            images = F.interpolate(images, size=(image_size, image_size),
                                   mode='bilinear', align_corners=False)
            masks  = F.interpolate(masks.unsqueeze(1).float(), size=(image_size, image_size),
                                   mode='nearest').squeeze(1).long()

            images = images.unsqueeze(2).repeat(1, 1, D_FRAMES, 1, 1)
            masks  = masks.unsqueeze(1).repeat(1, D_FRAMES, 1, 1)

            logits = model(images)
            loss   = criterion(logits, masks)
            preds  = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            for c in range(num_classes):
                class_dices[c].append(dice_score(preds, masks, class_idx=c))

    class_dice_means = {c: float(np.mean(v)) for c, v in class_dices.items()}
    mean_tumor       = _mean_tumor_dice(class_dice_means)
    return total_loss / len(loader), class_dice_means, mean_tumor


def validate(model, loader, criterion, device, num_classes=8, image_size=64):
    return evaluate(model, loader, criterion, device, num_classes, image_size, split='val')


def main():
    global D_FRAMES  # may be overridden by --d-frames or --config
    parser = argparse.ArgumentParser(description='Train 3D Attention U-Net')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-root', type=str,
                       default=str(Path(__file__).parent.parent / 'data' / 'raw' / 'Images_' / 'Images_'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--viz-dir', default='visualizations')
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes (0 = main process only, safe for all envs)')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Resize images to this spatial size before training (64 is fast on CPU, 128 for GPU)')
    parser.add_argument('--d-frames', type=int, default=D_FRAMES,
                        help='Pseudo-3D depth: slices stacked per sample (2=CPU, 4=8GB GPU, 8=16GB GPU)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON config file — overrides individual flags (see configs/)')

    args = parser.parse_args()

    # Load config file if provided — values override argparse defaults
    if args.config:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            attr = key.replace('-', '_')
            if hasattr(args, attr):
                setattr(args, attr, val)
            else:
                logger.warning(f"Config key '{key}' is not a known argument — skipping")

    # Apply d_frames from args (may have been overridden by config)
    D_FRAMES = args.d_frames

    # Setup
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint dir: {ckpt_dir}")
    logger.info(f"Visualization dir: {viz_dir}")

    # Create dataloaders
    data_root = resolve_data_root(args.data_root)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root,
        batch_size=args.batch,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = AttentionUNet3D(
        in_channels=1,
        num_classes=args.num_classes,
        base_filters=args.base_filters,
        depth=args.depth,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    # Class weights: inversely proportional to image count per category.
    # bg=0.01 (trivially easy), rare classes (Germ Cell, Mesenchymal) weighted highest.
    _weights_by_classes = {
        3: [0.01, 0.60, 0.39],
        4: [0.01, 0.40, 0.30, 0.29],
        8: [0.01, 0.18, 0.18, 0.15, 0.14, 0.14, 0.11, 0.09],
    }
    criterion = HybridLoss(
        num_classes=args.num_classes,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        dice_weights=_weights_by_classes.get(args.num_classes)
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Resume from checkpoint if requested
    best_val_dice = 0.0
    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_dice = float(checkpoint.get('val_dice', 0.0))
        start_epoch   = int(checkpoint.get('epoch', 0)) + 1
        logger.info(f"Resuming from checkpoint: {resume_path}")
        logger.info(f"  Epoch {start_epoch - 1} | best val dice {best_val_dice:.4f}")

        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch - 1} >= requested total {args.epochs}. "
                "Pass a larger --epochs value to continue training."
            )

    # Model config — saved in every checkpoint so predict3d.py can rebuild the model
    model_config = {
        'base_filters': args.base_filters,
        'depth': args.depth,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'd_frames': D_FRAMES,
    }

    # Training loop
    logger.info("Starting training...")
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    
    visualizer = SegmentationVisualizer(output_dir=str(viz_dir))
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch [{epoch}/{args.epochs}]")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_class_dices, train_mean = train_epoch(
            model, train_loader, criterion, optimizer, device, args.image_size)
        train_losses.append(train_loss)
        train_dices.append(train_mean)

        # Validate
        val_loss, val_class_dices, val_mean = validate(
            model, val_loader, criterion, device,
            num_classes=args.num_classes, image_size=args.image_size)
        val_losses.append(val_loss)
        val_dices.append(val_mean)

        # ── Epoch summary ──────────────────────────────────────────────
        def _class_str(dices):
            return ' | '.join(f"{CLASS_NAMES[c]}: {dices[c]:.4f}" for c in TUMOR_CLASSES)

        logger.info(f"Train Loss: {train_loss:.4f} | Mean Tumor Dice: {train_mean:.4f}")
        logger.info(f"  {_class_str(train_class_dices)}")
        logger.info(f"Val   Loss: {val_loss:.4f} | Mean Tumor Dice: {val_mean:.4f}")
        logger.info(f"  {_class_str(val_class_dices)} | Background: {val_class_dices[0]:.4f}")

        # Scheduler step
        scheduler.step(val_loss)

        def _make_ckpt(extra=None):
            d = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice':             val_mean,
                'val_class_dices':      val_class_dices,
                'val_loss':             val_loss,
                'train_loss':           train_loss,
                'model_config':         model_config,
            }
            if extra:
                d.update(extra)
            return d

        # Save best checkpoint
        if val_mean > best_val_dice:
            best_val_dice = val_mean
            ckpt_path = ckpt_dir / f'best_model_dice_{val_mean:.4f}.pt'
            torch.save(_make_ckpt(), ckpt_path)
            logger.info(f"✓ New best checkpoint: {ckpt_path}")

        # Save latest checkpoint every epoch (resume point after any shutdown)
        torch.save(_make_ckpt(), ckpt_dir / 'checkpoint_latest.pt')
        logger.info(f"✓ Saved latest checkpoint (epoch {epoch})")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(_make_ckpt(), ckpt_path)
            logger.info(f"✓ Saved periodic checkpoint: {ckpt_path}")
        
        # Visualize predictions on validation set
        if epoch % 5 == 0:
            logger.info("Generating validation visualizations...")
            try:
                # Get one batch for visualization
                val_batch = next(iter(val_loader))
                with torch.no_grad():
                    images = val_batch['image'].to(device)   # (B, 1, H, W)
                    masks = val_batch['mask'].to(device)     # (B, H, W)
                    images = F.interpolate(images, size=(args.image_size, args.image_size),
                                           mode='bilinear', align_corners=False)
                    masks = F.interpolate(masks.unsqueeze(1).float(),
                                          size=(args.image_size, args.image_size),
                                          mode='nearest').squeeze(1).long()
                    # Same 2D->pseudo-3D conversion used in train_epoch/validate
                    images_3d = images.unsqueeze(2).repeat(1, 1, D_FRAMES, 1, 1)
                    logits = model(images_3d)
                    preds = torch.argmax(logits, dim=1)      # (B, D_FRAMES, H, W)

                # Use original 2D image + middle slice of the pseudo-volume prediction
                mid = D_FRAMES // 2
                for i in range(min(2, len(images))):
                    img_np = images[i, 0].cpu().numpy()      # (H, W)
                    mask_np = masks[i].cpu().numpy()         # (H, W)
                    pred_np = preds[i, mid].cpu().numpy()    # middle slice
                    
                    # Normalize image
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    
                    fig = visualizer.visualize_segmentation_comparison(
                        image=img_np,
                        mask=mask_np,
                        prediction=pred_np,
                        title=f'Epoch {epoch} - Sample {i}',
                        save_path=str(viz_dir / f'epoch_{epoch:03d}_sample_{i}.png')
                    )
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error in visualization: {e}")
    
    # Plot training curves (loss + Dice per epoch, train vs val)
    logger.info("Plotting training curves...")
    fig = visualizer.plot_training_curves(
        train_losses, val_losses, train_dices, val_dices,
        save_path=str(viz_dir / 'training_curves.png')
    )
    plt.close(fig)
    logger.info(f"Training curves saved to: {viz_dir / 'training_curves.png'}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best Val Mean Tumor Dice: {best_val_dice:.4f}")
    logger.info(f"{'='*60}")

    # ── Test set evaluation ────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for test set evaluation...")
    best_ckpts = sorted(ckpt_dir.glob('best_model_dice_*.pt'),
                        key=lambda p: float(p.stem.split('_')[-1]), reverse=True)
    if best_ckpts:
        best_ckpt = best_ckpts[0]
        ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data['model_state_dict'])
        logger.info(f"Loaded: {best_ckpt.name}")

    logger.info("Running test set evaluation...")
    test_loss, test_class_dices, test_mean = evaluate(
        model, test_loader, criterion, device,
        num_classes=args.num_classes, image_size=args.image_size, split='test')

    per_class_str = ' | '.join(
        f"{CLASS_NAMES[c]}: {test_class_dices[c]:.4f}" for c in TUMOR_CLASSES
    )
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Tumor Dice: {test_mean:.4f}")
    logger.info(f"  {per_class_str}")
    logger.info(f"Background Dice: {test_class_dices[0]:.4f}")
    logger.info(f"Test Loss      : {test_loss:.4f}")
    logger.info(f"{'='*60}")

    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info(f"Visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    main()
