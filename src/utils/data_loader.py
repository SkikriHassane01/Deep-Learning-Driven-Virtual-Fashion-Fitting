"""Dataset handling and data loading utilities."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List
from datasets import load_dataset
import random

from .config import Config


class DataTransforms:
    """Data augmentation and preprocessing transforms"""
    
    @staticmethod
    def get_train_transforms(input_size: Tuple[int, int] = (512, 512)) -> A.Compose:
        """Training data augmentation pipeline"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_val_transforms(input_size: Tuple[int, int] = (512, 512)) -> A.Compose:
        """Validation data preprocessing pipeline"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_inference_transforms(input_size: Tuple[int, int] = (512, 512)) -> A.Compose:
        """Inference preprocessing pipeline"""
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class HumanParsingDataset(Dataset):
    """Dataset class for Human Parsing task"""
    
    def __init__(self, dataset, transform: Optional[A.Compose] = None, 
                 num_classes: int = 18, ignore_index: int = 255):
        """
        Initialize dataset.
        
        Args:
            dataset: HuggingFace dataset or similar
            transform: Albumentations transform pipeline
            num_classes: Number of parsing classes
            ignore_index: Index to ignore in loss calculation
        """
        self.dataset = dataset
        self.transform = transform
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        
        # Process image
        image = sample["image"]
        if not isinstance(image, np.ndarray):
            image = np.array(image.convert("RGB"))
        
        # Process mask
        mask = sample["mask"]
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Handle invalid mask values
        mask = mask.astype(np.int32)
        mask[mask < 0] = self.ignore_index
        mask[mask >= self.num_classes] = self.ignore_index
        mask = mask.astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].float()
            mask = transformed["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask


def create_data_loaders(dataset_name: str = "mattmdjaga/human_parsing_dataset",
                       split: str = "train",
                       batch_size: int = 10,
                       num_workers: int = 4,
                       train_split: float = 0.8,
                       input_size: Tuple[int, int] = (512, 512),
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_split: Fraction of data to use for training
        input_size: Input image size (height, width)
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Split into train and validation
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    
    split_point = int(train_split * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    # Create datasets
    train_dataset = HumanParsingDataset(
        dataset.select(train_indices),
        transform=DataTransforms.get_train_transforms(input_size)
    )
    
    val_dataset = HumanParsingDataset(
        dataset.select(val_indices),
        transform=DataTransforms.get_val_transforms(input_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    return train_loader, val_loader


def setup_data_loading(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Setup data loading using config parameters.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    return create_data_loaders(
        dataset_name=config.DATASET_NAME,
        split=config.SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        input_size=config.INPUT_SIZE,
        seed=config.SEED
    )