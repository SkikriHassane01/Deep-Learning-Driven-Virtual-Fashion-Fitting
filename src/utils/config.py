"""Centralized configuration management for the virtual fashion try-on system."""

import os
import torch
from typing import Tuple, Optional, List


class Config:
    """Centralized configuration for the Virtual Fashion Try-On system"""
    
    # Dataset Configuration
    DATASET_NAME: str = "mattmdjaga/human_parsing_dataset"
    SPLIT: str = "train"
    NUM_CLASSES: int = 18
    IGNORE_INDEX: int = 255
    
    # Image Configuration
    IMAGE_SIZE: Tuple[int, int] = (512, 512)  # Standard size for all operations
    INPUT_SIZE: Tuple[int, int] = (512, 512)  # Keeping for backward compatibility
    
    # Model Configuration
    BACKBONE: str = "resnet101"
    
    # Training Configuration
    BATCH_SIZE: int = 10
    EPOCHS: int = 5
    LEARNING_RATE_BACKBONE: float = 1e-4
    LEARNING_RATE_HEAD: float = 5e-4
    WEIGHT_DECAY: float = 5e-4
    GRADIENT_CLIP: float = 1.0
    
    # Loss Configuration
    EDGE_WEIGHT: float = 0.4
    
    # Paths
    MODEL_DIR: str = "model"
    OUTPUT_DIR: str = os.path.join(MODEL_DIR, "output")
    MODEL_PATH: str = os.path.join(MODEL_DIR, "best_model.pth")
    CONFIG_PATH: str = os.path.join(MODEL_DIR, "model_config.json")
    CHECKPOINT_DIR: str = os.path.join(MODEL_DIR, "checkpoints")
    MODEL_SAVE_PATH: str = os.path.join(MODEL_DIR, "best_model.pth")
    
    # System Configuration
    SEED: int = 42
    NUM_WORKERS: int = 10
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Resume Training
    RESUME_FROM: Optional[str] = None
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE: int = 5
    
    # Garment Generation Configuration
    DIFFUSION_MODEL: str = "runwayml/stable-diffusion-v1-5"
    INPAINT_MODEL: str = "stabilityai/stable-diffusion-2-inpainting"
    NUM_INFERENCE_STEPS: int = 25
    GUIDANCE_SCALE: float = 8.0
    
    # Virtual Try-On Configuration
    CLOTHING_CLASSES: List[int] = [4, 5, 7]  # Upper-clothes, Skirt, Dress
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for outputs"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
    
    @classmethod
    def update_device(cls, device: str = None):
        """Update device configuration"""
        if device:
            cls.DEVICE = torch.device(device)
        else:
            cls.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def update_paths(cls, base_dir: str):
        """Update all paths based on new base directory"""
        cls.MODEL_DIR = os.path.join(base_dir, "model")
        cls.OUTPUT_DIR = os.path.join(cls.MODEL_DIR, "output")
        cls.MODEL_PATH = os.path.join(cls.MODEL_DIR, "best_model.pth")
        cls.CONFIG_PATH = os.path.join(cls.MODEL_DIR, "model_config.json")
        cls.CHECKPOINT_DIR = os.path.join(cls.MODEL_DIR, "checkpoints")


# Class names and colors for human parsing
CLASS_NAMES = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face",
    "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"
]

CLASS_COLORS = [
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51], [255, 85, 0],
    [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85], [85, 51, 0], [52, 86, 128],
    [0, 128, 0], [0, 0, 255], [51, 170, 221], [0, 255, 255], [85, 255, 170], [170, 255, 85]
]

# Add alias for backward compatibility
COLORS = CLASS_COLORS