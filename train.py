#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from models.tryon_model import train_model

def validate_data_structure(data_dir):
    """Validate that the data directory has the correct structure"""
    data_dir = Path(data_dir)
    required_files = ["train_pairs.txt", "test_pairs.txt"]
    
    print("Validating data structure...")
    
    # Check required files
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise ValueError(f"Required file not found: {file_path}")
        
        # Check if file has content
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not lines:
            raise ValueError(f"File {file_path} is empty or has no valid pairs")
        
        print(f"✅ {file_name}: {len(lines)} pairs found")
    
    # Check for your specific directory structure: train/image/, test/image/, train/cloth/, test/cloth/
    required_dirs = [
        ("train/image", "Train person images"),
        ("train/cloth", "Train clothing images"),
        ("test/image", "Test person images"),
        ("test/cloth", "Test clothing images")
    ]
    
    for dir_name, description in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            raise ValueError(f"Required directory not found: {dir_path} ({description})")
        
        # Count images
        image_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        print(f"✅ {dir_name}: {len(image_files)} images found")
    
    # Check optional directories
    optional_dirs = [
        ("train/cloth-mask", "Train clothing masks"),
        ("test/cloth-mask", "Test clothing masks"),
        ("train/openpose_json", "Train pose data"),
        ("test/openpose_json", "Test pose data")
    ]
    
    for dir_name, description in optional_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"✅ {dir_name}: {len(files)} files found ({description})")
        else:
            print(f"⚠️  {dir_name}: Not found ({description}) - will use defaults")
    
    print("Data structure validation passed!")

def main():
    parser = argparse.ArgumentParser(description="Train virtual try-on model")
    parser.add_argument("--data-dir", type=str, default="./data", 
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Path to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate data structure, don't train")
    
    args = parser.parse_args()
    
    try:
        # Validate data directory structure
        validate_data_structure(args.data_dir)
        
        if args.validate_only:
            print("Data validation completed successfully!")
            return
        
        print("Starting Virtual Try-On Model Training")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        
        # Train the model
        best_model_path = train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_checkpoint=args.resume
        )
        
        print(f"Training completed! Best model saved at: {best_model_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease check:")
        print("1. Data directory structure")
        print("2. Train/test pairs files")
        print("3. Image files in person/cloth directories")
        return 1

if __name__ == "__main__":
    exit(main())
