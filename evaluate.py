#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(pred_path, target_path):
    """Calculate image quality metrics between prediction and target."""
    # Load images
    pred = np.array(Image.open(pred_path).convert('RGB'))
    target = np.array(Image.open(target_path).convert('RGB'))
    
    # Ensure same dimensions
    if pred.shape != target.shape:
        # Resize prediction to match target
        pred = np.array(Image.fromarray(pred).resize((target.shape[1], target.shape[0])))
    
    # Calculate PSNR
    psnr_value = psnr(target, pred, data_range=255)
    
    # Calculate SSIM
    ssim_value = ssim(target, pred, multichannel=True, data_range=255, channel_axis=2)
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value
    }

def evaluate_model(model_path, test_data_dir, output_dir):
    """Evaluate the virtual try-on model on the test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Auto-detect directory structure
    test_data_dir = Path(test_data_dir)
    print(f"🔍 Checking test data structure in: {test_data_dir}")
    
    # Look for common directory patterns
    possible_person_dirs = ["image", "person", "test/image"]
    possible_cloth_dirs = ["cloth", "clothing", "test/cloth"]
    
    person_dir = None
    cloth_dir = None
    
    for dir_name in possible_person_dirs:
        check_dir = test_data_dir / dir_name
        if check_dir.exists() and list(check_dir.glob("*.jpg")):
            person_dir = check_dir
            print(f"✅ Found person images in: {dir_name}")
            break
    
    for dir_name in possible_cloth_dirs:
        check_dir = test_data_dir / dir_name
        if check_dir.exists() and list(check_dir.glob("*.jpg")):
            cloth_dir = check_dir
            print(f"✅ Found cloth images in: {dir_name}")
            break
    
    if not person_dir or not cloth_dir:
        print("❌ Could not find person or cloth directories with images")
        print(f"Looked for person images in: {possible_person_dirs}")
        print(f"Looked for cloth images in: {possible_cloth_dirs}")
        return None
    
    # Try to load model
    try:
        from models.tryon_model import VirtualTryOnModel
        model = VirtualTryOnModel(model_path=model_path, device=device)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Proceeding with baseline evaluation...")
        model = None
    
    # Load test data
    test_data_dir = Path(test_data_dir)
    test_pairs_file = test_data_dir / 'test_pairs.txt'
    
    # Use data directory test_pairs if exists, otherwise look in test_data_dir
    if not test_pairs_file.exists():
        # Try to find it in the parent data dir
        data_dir = test_data_dir.parent
        test_pairs_file = data_dir / 'test_pairs.txt'
        
    if not test_pairs_file.exists():
        raise FileNotFoundError(f"Test pairs file not found at {test_pairs_file} or parent directory")
    
    # Read test pairs
    test_pairs = []
    with open(test_pairs_file, 'r') as f:
        for line in f:
            if line.strip():
                person_img, cloth_img = line.strip().split()
                test_pairs.append((person_img, cloth_img))
    
    print(f"Found {len(test_pairs)} test pairs")
    
    # Initialize metrics
    metrics = {
        'psnr': [],
        'ssim': []
    }
    
    # Process each test pair
    successful_pairs = 0
    for i, (person_img, cloth_img) in enumerate(tqdm(test_pairs)):
        # Construct paths - fix directory structure
        person_path = test_data_dir / 'image' / person_img  # Changed from 'person' to 'image'
        cloth_path = test_data_dir / 'cloth' / cloth_img
        cloth_mask_path = test_data_dir / 'cloth-mask' / cloth_img
        
        # Get person name without extension for pose
        person_name = os.path.splitext(person_img)[0]
        pose_path = test_data_dir / 'openpose_json' / f"{person_name}_keypoints.json"  # Added '_keypoints'
        
        # Debug: Print what we're looking for
        if i < 5:  # Only print first 5 for debugging
            print(f"\nDebug pair {i+1}:")
            print(f"  Person: {person_path} (exists: {person_path.exists()})")
            print(f"  Cloth: {cloth_path} (exists: {cloth_path.exists()})")
            print(f"  Mask: {cloth_mask_path} (exists: {cloth_mask_path.exists()})")
            print(f"  Pose: {pose_path} (exists: {pose_path.exists()})")
        
        # Skip if any required file is missing
        required_files = [person_path, cloth_path]
        if not all(p.exists() for p in required_files):
            if i < 10:  # Only print first 10 missing files
                print(f"Skipping pair {i+1}: {person_img}, {cloth_img} due to missing files")
            continue
        
        # Generate try-on result
        try:
            if model is not None:
                result_img = model.try_on(
                    person_path, cloth_path, pose_path, 
                    cloth_mask_path if cloth_mask_path.exists() else None
                )
            else:
                # Use baseline method
                from tryon.services.fitter import baseline_fit
                from tryon.services.pose_seg import process_user_image
                
                # Create temporary output directory
                temp_dir = output_dir / f"temp_{i}"
                temp_dir.mkdir(exist_ok=True)
                
                # Process user image if pose doesn't exist
                if not pose_path.exists():
                    _, pose_path = process_user_image(person_path, temp_dir)
                
                result_path = temp_dir / "result.jpg"
                baseline_fit(
                    person_path, pose_path, cloth_path, 
                    cloth_mask_path if cloth_mask_path.exists() else None,
                    result_path
                )
                result_img = Image.open(result_path)
            
            # Save result
            result_path = output_dir / f"result_{i+1:04d}.jpg"
            result_img.save(result_path)
            
            # Save reference images for comparison
            person_out_path = output_dir / f"person_{i+1:04d}.jpg"
            cloth_out_path = output_dir / f"cloth_{i+1:04d}.jpg"
            shutil.copy2(person_path, person_out_path)
            shutil.copy2(cloth_path, cloth_out_path)
            
            # Calculate metrics (using person image as reference for now)
            pair_metrics = calculate_metrics(result_path, person_path)
            
            # Add metrics to overall results
            for metric_name, value in pair_metrics.items():
                metrics[metric_name].append(value)
            
            successful_pairs += 1
            
        except Exception as e:
            print(f"Error processing pair {i+1}: {e}")
            continue
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    
    # Print results
    print(f"\nEvaluation Results ({successful_pairs}/{len(test_pairs)} successful):")
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    
    # Save metrics to file
    with open(output_dir / "metrics.txt", 'w') as f:
        f.write(f"Successful pairs: {successful_pairs}/{len(test_pairs)}\n")
        for metric_name, value in avg_metrics.items():
            f.write(f"{metric_name}: {value:.6f}\n")
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    return avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate virtual try-on model")
    parser.add_argument('--model-path', type=str, help='Path to the model checkpoint (optional)')
    parser.add_argument('--test-dir', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Warning: Model file {args.model_path} not found. Using baseline method.")
        args.model_path = None
    
    evaluate_model(args.model_path, args.test_dir, args.output_dir)
