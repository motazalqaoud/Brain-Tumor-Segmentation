"""
brain_tumor_loader.py
---------------------
Data loader for Kaggle Brain Tumor MRI dataset.

Handles:
- JSON metadata parsing (tumor type, bounding boxes, masks)
- Image loading (PNG/JPG MRI slices)
- Mask and metadata association
- Train/val/test splitting
- 3D volume assembly (if needed)

Dataset structure:
    data/raw/Images_/
    ├── [Tumor Type]/
    │   ├── [MRI Modality]/
    │   │   ├── [Tumor Subtype]/
    │   │   │   ├── image.jpg
    │   │   │   ├── image_mask_consensus.png
    │   │   │   ├── image_bbox.png
    │   │   │   └── image_meta.json

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    This dataset contains MRI slices (not full 3D volumes).
    For 3D training, we assemble adjacent slices into volumes.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


class BrainTumorDataset(Dataset):
    """
    Kaggle Brain Tumor MRI Dataset.
    
    Loads 2D slices and optionally assembles them into 3D volumes.
    """
    
    # Tumor type to class mapping
    TUMOR_CLASSES = {
        'Glioma': 1,
        'Meningioma': 2,
        'Pituitary': 3,
    }
    
    # MRI modalities
    MODALITIES = ['T1', 'T1c', 'T2', 'FLAIR']
    
    def __init__(self, root_dir: str, modality: str = 'T1c', 
                 mode: str = 'slice', split: str = 'train',
                 transform=None, volume_size: int = 128):
        """
        Args:
            root_dir: path to data/raw/Images_/
            modality: which MRI modality to use ('T1', 'T1c', 'T2', 'FLAIR')
            mode: 'slice' (2D) or 'volume' (3D assembled)
            split: 'train', 'val', or 'test'
            transform: image transformations
            volume_size: target size for resizing (default 128)
        """
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.mode = mode
        self.split = split
        self.transform = transform
        self.volume_size = volume_size
        
        self.images = []
        self.metadata = []
        self.tumor_types = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Scan directory and load all image paths with metadata."""
        logger.info(f"Loading {self.modality} images from {self.root_dir}")
        
        # Scan all subdirectories
        image_count = 0
        for tumor_type_dir in self.root_dir.iterdir():
            if not tumor_type_dir.is_dir():
                continue
            
            tumor_type = tumor_type_dir.name
            
            # Look for modality subdirs (modality is in the folder name like "T1C+", "T2", etc.)
            for modality_dir in tumor_type_dir.iterdir():
                if not modality_dir.is_dir():
                    continue
                
                # Check if modality matches (handle "T1C+", "T1", "T2", "FLAIR")
                modality_name = modality_dir.name.lower()
                modality_query = self.modality.lower()
                
                if modality_query not in modality_name:
                    continue
                
                # Scan subtype directories
                for subtype_dir in modality_dir.iterdir():
                    if not subtype_dir.is_dir():
                        continue
                    
                    # Find image files
                    for file in subtype_dir.iterdir():
                        if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            if '_mask' in file.name or '_bbox' in file.name:
                                continue
                            if file.name.endswith('.meta.json'):
                                continue
                            
                            # Find corresponding mask
                            mask_file = file.parent / f"{file.stem}_mask_consensus.png"
                            meta_file = file.parent / f"{file.stem}_meta.json"
                            
                            if mask_file.exists():
                                self.images.append(str(file))
                                
                                meta = {}
                                if meta_file.exists():
                                    try:
                                        with open(meta_file) as f:
                                            meta = json.load(f)
                                    except:
                                        pass
                                
                                meta['mask_path'] = str(mask_file)
                                meta['tumor_type'] = tumor_type
                                meta['subtype'] = subtype_dir.name
                                
                                self.metadata.append(meta)
                                self.tumor_types.append(tumor_type)
                                image_count += 1
        
        logger.info(f"Loaded {image_count} images")
        
        # Compute train/val/test split
        if self.split != 'all':
            self._split_dataset()
    
    def _split_dataset(self):
        """Split into train/val/test (70/15/15)."""
        n = len(self.images)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)
        
        if self.split == 'train':
            indices = indices[:train_end]
        elif self.split == 'val':
            indices = indices[train_end:val_end]
        else:  # test
            indices = indices[val_end:]
        
        self.images = [self.images[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]
        self.tumor_types = [self.tumor_types[i] for i in indices]
        
        logger.info(f"Split {self.split}: {len(self.images)} images")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load image and mask.
        
        Returns:
            dict with 'image', 'mask', 'label', 'metadata'
        """
        img_path = self.images[idx]
        meta = self.metadata[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        image = image / 255.0  # Normalize to [0, 1]
        
        # Resize
        image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize(
            (self.volume_size, self.volume_size), Image.BILINEAR
        ), dtype=np.float32) / 255.0
        
        # Load mask
        mask_path = meta['mask_path']
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = mask / 255.0
        
        # Resize mask
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.volume_size, self.volume_size), Image.NEAREST
        ), dtype=np.float32) / 255.0
        
        # Binary mask (threshold)
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W) with values 0 or 1
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Get tumor class
        tumor_type = meta.get('tumor_type', 'Unknown')
        label = self.TUMOR_CLASSES.get(tumor_type, 0)
        
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'tumor_type': tumor_type,
            'image_path': img_path,
            'metadata': meta
        }


def load_brain_tumor_dataset(root_dir: str, modality: str = 'T1c',
                            batch_size: int = 4, num_workers: int = 2,
                            split: str = 'train') -> DataLoader:
    """
    Convenience function to load brain tumor dataset.
    
    Args:
        root_dir: path to data/raw/Images_/
        modality: MRI modality
        batch_size: batch size
        num_workers: number of dataloader workers
        split: 'train', 'val', or 'test'
    
    Returns:
        PyTorch DataLoader
    """
    dataset = BrainTumorDataset(
        root_dir=root_dir,
        modality=modality,
        split=split,
        mode='slice'
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        pin_memory=True
    )


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    dataset = BrainTumorDataset(
        root_dir=str(Path(__file__).parent.parent.parent / 'data' / 'raw' / 'Images_'),
        modality='T1c',
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Tumor type: {sample['tumor_type']}")
    print(f"Label: {sample['label']}")
