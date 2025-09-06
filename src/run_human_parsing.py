#!/usr/bin/env python3
"""
Human Parsing Pipeline Entry Point
Simple script that provides options to train, resume training, or run inference using existing modules
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Add src to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from models.human_parser import HumanParsingNet
from utils.config import Config, CLASS_NAMES, CLASS_COLORS
from utils.visualization import visualize_predictions, save_visualization
from utils.image_processing import preprocess_image, postprocess_prediction
from training.trainer import Trainer
from utils.data_loader import create_data_loaders


def download_test_image(url=None):
    """Download test image from the internet"""
    if url is None:
        # Default test image URLs
        test_urls = [
            "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=600",  # Man portrait
            "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=600",  # Woman fashion
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600",  # Person standing
        ]
        
        print("\nAvailable test images:")
        for i, test_url in enumerate(test_urls, 1):
            print(f"{i}. {test_url}")
        
        choice = input(f"\nSelect image (1-{len(test_urls)}) or enter custom URL: ").strip()
        
        if choice.startswith('http'):
            url = choice
        elif choice.isdigit() and 1 <= int(choice) <= len(test_urls):
            url = test_urls[int(choice) - 1]
        else:
            url = test_urls[0]  # Default
    
    try:
        print(f"📥 Downloading image from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Load image
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Save locally
        os.makedirs("outputs/human_parsing", exist_ok=True)
        image_path = "outputs/human_parsing/downloaded_test_image.jpg"
        image.save(image_path)
        
        print(f"✅ Image downloaded and saved: {image_path}")
        return image, image_path
        
    except Exception as e:
        print(f"❌ Failed to download image: {e}")
        return None, None


def create_simple_test_image(size):
    """Create a simple test image for demonstration"""
    width, height = size if isinstance(size, tuple) else (size, size)
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background
    img_array[:, :] = [100, 150, 200]
    
    # Simple person silhouette
    # Head region
    y_start, y_end = int(height * 0.1), int(height * 0.3)
    x_start, x_end = int(width * 0.4), int(width * 0.6)
    img_array[y_start:y_end, x_start:x_end] = [200, 180, 160]
    
    # Torso region
    y_start, y_end = int(height * 0.3), int(height * 0.7)
    x_start, x_end = int(width * 0.35), int(width * 0.65)
    img_array[y_start:y_end, x_start:x_end] = [150, 100, 80]
    
    return Image.fromarray(img_array)


def run_inference_mode(config):
    """Run inference using pretrained model"""
    print("\n" + "="*50)
    print("🔍 INFERENCE MODE")
    print("="*50)
    
    # Initialize model
    print("🔄 Initializing Human Parsing Model...")
    model = HumanParsingNet(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    # Load pretrained weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"📁 Loading pretrained weights from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Pretrained weights loaded successfully!")
    else:
        print(f"⚠️ No pretrained weights found at {config.MODEL_SAVE_PATH}")
        print("Using initialized model without trained weights")
    
    model.eval()
    
    # Choose test image source
    print("\nSelect test image source:")
    print("1. Download from internet")
    print("2. Use simple generated test image")
    
    choice = input("Choose option (1-2): ").strip()
    
    if choice == "1":
        test_image, image_path = download_test_image()
        if test_image is None:
            print("Falling back to simple test image...")
            test_image = create_simple_test_image(config.IMAGE_SIZE)
    else:
        test_image = create_simple_test_image(config.IMAGE_SIZE)
    
    # Resize to model input size
    test_image = test_image.resize(config.IMAGE_SIZE)
    
    print(f"\n🖼️ Processing test image...")
    
    # Preprocess image
    image_tensor = preprocess_image(test_image, config.IMAGE_SIZE, config.DEVICE)
    
    # Run inference
    print("🔍 Running human parsing inference...")
    with torch.no_grad():
        prediction = model(image_tensor.unsqueeze(0))
        if isinstance(prediction, tuple):
            prediction = prediction[1]  # Use refined prediction
        prediction = prediction.squeeze(0)
    
    # Postprocess prediction
    pred_mask = postprocess_prediction(prediction)
    
    # Visualize results
    print("📊 Creating visualization...")
    fig = visualize_predictions(
        test_image, 
        pred_mask, 
        CLASS_NAMES[:config.NUM_CLASSES],
        CLASS_COLORS[:config.NUM_CLASSES]
    )
    
    # Save results
    output_path = "outputs/human_parsing/inference_result.png"
    save_visualization(fig, output_path)
    
    print(f"✅ Results saved to: {output_path}")
    
    # Show class distribution
    unique, counts = np.unique(pred_mask, return_counts=True)
    print(f"\n📈 Detected classes:")
    for class_id, count in zip(unique, counts):
        if class_id < len(CLASS_NAMES):
            print(f"  - {CLASS_NAMES[class_id]}: {count} pixels ({count/pred_mask.size*100:.1f}%)")


def run_training_mode(config, resume=False):
    """Run training mode (fresh start or resume)"""
    mode_name = "RESUME TRAINING" if resume else "FRESH TRAINING"
    print(f"\n" + "="*50)
    print(f"🎯 {mode_name} MODE")
    print("="*50)
    
    try:
        # Create data loaders
        print("📊 Creating data loaders...")
        train_loader, val_loader = create_data_loaders()
        print(f"✅ Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
        
        # Initialize model
        print("🔄 Initializing model...")
        model = HumanParsingNet(num_classes=config.NUM_CLASSES)
        model = model.to(config.DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Initialize trainer
        trainer = Trainer(model, train_loader, val_loader, config)
        
        if resume and os.path.exists(config.CHECKPOINT_DIR):
            # Find latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(config.CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print(f"📁 Resuming from: {latest_checkpoint}")
                # Set resume path in config
                config.RESUME_FROM = latest_checkpoint
                trainer = Trainer(model, train_loader, val_loader, config)
        
        # Start training
        print(f"🚀 Starting {'resumed' if resume else 'fresh'} training...")
        best_miou = trainer.train()
        
        print(f"\n✅ Training completed! Best mIoU: {best_miou:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This might be due to missing dataset or dependencies.")


def show_menu():
    """Display main menu options"""
    print("\n" + "="*60)
    print("🧑 HUMAN PARSING PIPELINE")
    print("="*60)
    print("Choose an option:")
    print("1. 🔍 Run Inference (using pretrained model)")
    print("2. 🎯 Train Model (fresh start)")
    print("3. ▶️  Resume Training (from checkpoint)")
    print("4. ❌ Exit")
    print("="*60)


def main():
    """Main execution pipeline for human parsing"""
    
    # Configuration
    config = Config()
    print(f"Device: {config.DEVICE}")
    print(f"Image Size: {config.IMAGE_SIZE}")
    print(f"Model Path: {config.MODEL_SAVE_PATH}")
    
    # Create output directories
    os.makedirs("outputs/human_parsing", exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            run_inference_mode(config)
            
        elif choice == "2":
            confirm = input("\n⚠️ This will start fresh training. Continue? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_training_mode(config, resume=False)
            else:
                print("Training cancelled.")
                
        elif choice == "3":
            confirm = input("\n⚠️ This will resume training from checkpoint. Continue? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_training_mode(config, resume=True)
            else:
                print("Resume cancelled.")
                
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice. Please select 1-4.")
        
        # Ask if user wants to continue
        if choice in ["1", "2", "3"]:
            continue_choice = input("\n🔄 Return to main menu? (Y/n): ").strip().lower()
            if continue_choice in ['n', 'no']:
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    main()