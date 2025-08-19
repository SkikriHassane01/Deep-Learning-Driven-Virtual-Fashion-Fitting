#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def setup_model_directories():
    """
    Create the correct directory structure for models and ensure
    the trained model is in the right location.
    """
    # Base paths
    project_root = Path(__file__).parent
    checkpoints_dir = project_root / "checkpoints"
    models_dir = project_root / "models"
    
    # Create needed directories
    os.makedirs(models_dir / "pose", exist_ok=True)
    os.makedirs(models_dir / "segmentation", exist_ok=True)
    os.makedirs(models_dir / "tryon", exist_ok=True)
    
    # Check if model exists in checkpoints but not in models/tryon
    source_model_paths = [
        checkpoints_dir / "best_model.pth",
        checkpoints_dir / "checkpoint_epoch_50.pth",  # Try final epoch
        # Check for other common checkpoint names
        *sorted(list(checkpoints_dir.glob("checkpoint_epoch_*.pth")), reverse=True)
    ]
    
    # Find first available checkpoint
    source_model = None
    for path in source_model_paths:
        if path.exists():
            source_model = path
            break
    
    target_model = models_dir / "tryon" / "best_model.pth"
    
    if source_model and source_model.exists() and not target_model.exists():
        print(f"Copying model from {source_model} to {target_model}")
        shutil.copy2(source_model, target_model)
        print("Model copied successfully")
    elif target_model.exists():
        print(f"Model already exists at {target_model}")
    else:
        print(f"Warning: No model found in checkpoints directory")
        print("Please place your trained model at one of these locations:")
        for path in source_model_paths[:3]:  # Show only first few paths
            print(f"- {path}")
    
    # Display paths for verification
    print("\nModel Search Paths:")
    print(f"- {checkpoints_dir / 'best_model.pth'}")
    print(f"- {models_dir / 'tryon' / 'best_model.pth'}")
    print(f"- {models_dir / 'best_model.pth'}")

if __name__ == "__main__":
    setup_model_directories()
    print("\nDirectory structure setup complete!")
