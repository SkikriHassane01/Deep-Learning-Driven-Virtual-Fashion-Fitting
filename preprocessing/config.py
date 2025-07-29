"""
Central configuration file for our project, its includes:
    - Centralized parameters: stores all important settings in one place
    - Ensure all scripts use the same parameters
    - make it easier to adjust settings without modifying multiple files 
"""

## import libraries
from pathlib import Path
import random
import numpy as np
import torch


## -------------- Data & Pose Directory --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
DEEP_FASHION_DIR = DATA_DIR / "DeepFashion" / "Images"
FASHION_GEN_DIR = DATA_DIR / "FashionGen" / "images"
ANNOTATIONS_DIR = DATA_DIR / "DeepFashion" / "annotations"
STYLES_GEN_DIR = DATA_DIR / "FashionGen" / "styles.csv"

CLEAN_DIR = ROOT_DIR / "Dataset_Clean"
POSE_DIR  = ROOT_DIR / "Pose"
## -------------- Preprocessing Parameters --------------------------------------------------
IMG_SIZE = 256
MIN_DETECTION_CONFIDENCE = 0.5
SEED = 42

## -------------- Augmentation Parameters --------------------------------------------------
FLIP_PROBABILITY: 0.5
BRIGHTNESS_RANGE = [-30, 30]
ROTATION_RANGE = [-15, 15]

## --------------- Sets Fixed Seed---------------------------------
def set_seeds(seed: int=SEED):
    """
    Sets fixed seeds for all random operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seeds()