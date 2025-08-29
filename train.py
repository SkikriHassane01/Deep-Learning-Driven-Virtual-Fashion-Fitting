#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path
from models.tryon_model import train_model

def read_pairs(pairs_path):
    """Read and validate pairs from a pairs file"""
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in {pairs_path}: {line}")
            pairs.append((parts[0], parts[1]))
    return pairs

def validate_data_structure(data_dir):
    """Validate that the data directory has the correct structure"""
    data_dir = Path(data_dir)
    required_files = ["train_pairs.txt", "test_pairs.txt"]
    
    print("📊 Validating data structure...")
    
    # Check required files
    ok = True
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"❌ Missing file: {file_path}")
            ok = False
            continue
        
        try:
            # Check pairs in files
            pairs = read_pairs(file_path)
            if not pairs:
                print(f"❌ No valid pairs in {file_path}")
                ok = False
            else:
                print(f"✅ {file_name}: {len(pairs)} pairs found")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            ok = False
    
    if not ok:
        return False
    
    # Check for required directory structure
    required_dirs = [
        ("train/image", "Train person images"),
        ("train/cloth", "Train clothing images"),
        ("test/image", "Test person images"),
        ("test/cloth", "Test clothing images"),
        ("train/cloth-mask", "Train clothing masks"),
        ("test/cloth-mask", "Test clothing masks"),
        ("train/openpose_json", "Train pose data"),
        ("test/openpose_json", "Test pose data")
    ]
    
    for dir_name, description in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_path} ({description})")
            ok = False
            continue
        
        # Count images/files in directory
        files = list(dir_path.glob("*.*"))
        print(f"✅ {dir_name}: {len(files)} files found ({description})")
    
    # Validate sample pairs exist
    for split, file_name in [("train", "train_pairs.txt"), ("test", "test_pairs.txt")]:
        pairs = read_pairs(data_dir / file_name)
        # Check first 5 pairs
        for i, (person, cloth) in enumerate(pairs[:5]):
            person_path = data_dir / split / "image" / person
            cloth_path = data_dir / split / "cloth" / cloth
            
            if not person_path.exists():
                print(f"❌ Person image not found: {person_path}")
                ok = False
            if not cloth_path.exists():
                print(f"❌ Cloth image not found: {cloth_path}")
                ok = False
    
    if ok:
        print("✅ Data structure validation passed!")
    else:
        print("❌ Data validation failed")
    
    return ok

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
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--cache-data", action="store_true",
                        help="Cache dataset in memory for faster training")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Set torch.backends.cudnn.benchmark=True for potentially faster training")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    # Performance optimization options
    parser.add_argument("--mixed-precision", action="store_true", 
                        help="Use mixed precision training for faster training")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Number of batches to accumulate gradients")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Number of warmup epochs")
    parser.add_argument("--cache-images", action="store_true",
                        help="Cache images in RAM for faster training")
    parser.add_argument("--amp-level", type=str, default="O1",
                        choices=["O0", "O1", "O2", "O3"],
                        help="AMP optimization level")
    parser.add_argument("--profile", action="store_true",
                        help="Run profiler for performance analysis")
    
    args = parser.parse_args()
    
    try:
        # Validate data directory structure
        if not validate_data_structure(args.data_dir):
            print("\nPlease check:")
            print("1. Data directory structure")
            print("2. Train/test pairs files")
            print("3. Image files in person/cloth directories")
            return 1
        
        if args.validate_only:
            print("✅ Validation-only mode completed successfully!")
            return 0
        
        print("\n🏋️ Starting Virtual Try-On Model Training")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size} (x{args.gradient_accumulation} accumulation = {args.batch_size * args.gradient_accumulation} effective)")
        print(f"Learning rate: {args.lr}")
        print(f"Mixed precision: {'✓' if args.mixed_precision else '✗'}")
        print(f"Number of workers: {args.num_workers}")
        print(f"Early stopping patience: {args.early_stopping if args.early_stopping > 0 else 'disabled'}")
        
        # Train the model
        best_model_path = train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_checkpoint=args.resume,
            num_workers=args.num_workers,
            cache_data=args.cache_data,
            benchmark=args.benchmark,
            save_every=args.save_every
        )
        
        print(f"\n🎉 Training completed! Best model saved at: {best_model_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease check:")
        print("1. Data directory structure")
        print("2. Train/test pairs files")
        print("3. Image files in person/cloth directories")
        return 1

if __name__ == "__main__":
    sys.exit(main())
