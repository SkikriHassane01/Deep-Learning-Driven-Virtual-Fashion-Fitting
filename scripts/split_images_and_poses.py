import os
import json
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import mediapipe as mp
import cv2
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
CSV_FILE = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Dataset_Clean\DeepFashion\annotation\DeepFashion.csv"
SOURCE_IMAGES = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Dataset_Clean\DeepFashion\Images"
OUTPUT_DIR = r"data/DeepFashion"
MAX_SAMPLES = None  # Set to a number for testing, None for all

# Helper functions
def find_image(source_dir, file_path):
    """Find image with multiple fallback options"""
    # Try direct path first
    direct_path = Path(source_dir) / file_path
    if direct_path.exists():
        return direct_path
        
    # Try with/without img/ prefix
    if file_path.startswith('img/'):
        alt_path = Path(source_dir) / file_path[4:]
        if alt_path.exists():
            return alt_path
    else:
        alt_path = Path(source_dir) / f"img/{file_path}"
        if alt_path.exists():
            return alt_path
    
    # Check parent directory
    img_dir = Path(source_dir) / 'img'
    if img_dir.exists():
        img_path = img_dir / file_path.replace('img/', '')
        if img_path.exists():
            return img_path
    
    return None

def create_better_mask(image):
    """Create a better binary mask by removing background"""
    # Convert to OpenCV format
    img_array = np.array(image)
    
    # Convert to grayscale if it's not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Return as PIL Image
    return Image.fromarray(mask)

def process_batch(df_batch):
    """Process a batch of records"""
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=1)
    
    found_count = 0
    
    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc="Processing"):
        try:
            # Get paths
            split = row['split']
            file_path = row['file']
            
            # Find source image
            source_img = find_image(SOURCE_IMAGES, file_path)
            if not source_img:
                continue
                
            # Get file paths maintaining directory structure
            dest_img = Path(OUTPUT_DIR) / split / 'image' / file_path
            pose_file = Path(OUTPUT_DIR) / split / 'pose' / Path(file_path).with_suffix('.json')
            cloth_file = Path(OUTPUT_DIR) / split / 'cloth' / Path(file_path).with_suffix('.png')
            mask_file = Path(OUTPUT_DIR) / split / 'cloth-mask' / Path(file_path).with_suffix('.png')
            parse_file = Path(OUTPUT_DIR) / split / 'image-parse' / Path(file_path).with_suffix('.png')
            
            # Create directories
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            pose_file.parent.mkdir(parents=True, exist_ok=True)
            cloth_file.parent.mkdir(parents=True, exist_ok=True)
            mask_file.parent.mkdir(parents=True, exist_ok=True)
            parse_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            shutil.copy2(source_img, dest_img)
            found_count += 1
            
            # Generate pose
            try:
                img = Image.open(dest_img).convert('RGB')
                results = pose_model.process(np.array(img))
                
                # Extract landmarks
                landmarks = []
                if results.pose_landmarks:
                    for lm in results.pose_landmarks.landmark:
                        landmarks.append({
                            'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility
                        })
                
                # Save pose
                with open(pose_file, 'w') as f:
                    json.dump({'landmarks': landmarks}, f)
                
                # Crop garment using bounding box
                x1, y1, x2, y2 = map(int, (row['x_1'], row['y_1'], row['x_2'], row['y_2']))
                
                # Validate coordinates
                width, height = img.size
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(x1+1, min(x2, width))
                y2 = max(y1+1, min(y2, height))
                
                # Crop and save
                crop = img.crop((x1, y1, x2, y2))
                crop.save(cloth_file)
                
                # Create better mask - FIX: Use image processing to create a proper binary mask
                mask = create_better_mask(crop)
                mask.save(mask_file)
                
                # Generate parsing map - FIX: Create a simple parsing map
                try:
                    # Simple colorization based on body parts detected by MediaPipe
                    parse_img = Image.new('L', img.size, 0)  # Start with black background
                    if results.pose_landmarks:
                        # Draw body parts in different gray values
                        parse_array = np.array(parse_img)
                        landmarks = results.pose_landmarks.landmark
                        
                        # Draw torso
                        cv2.rectangle(
                            parse_array,
                            (int(landmarks[11].x * width), int(landmarks[11].y * height)),
                            (int(landmarks[24].x * width), int(landmarks[24].y * height)),
                            120, -1  # Fill with gray value
                        )
                        
                        # Draw head area
                        cv2.circle(
                            parse_array,
                            (int(landmarks[0].x * width), int(landmarks[0].y * height)),
                            int(width * 0.1),  # 10% of image width as radius
                            80, -1  # Fill with darker gray
                        )
                        
                        # Draw arms
                        for i in range(11, 17):  # Left arm
                            if i < len(landmarks)-1:
                                cv2.line(
                                    parse_array,
                                    (int(landmarks[i].x * width), int(landmarks[i].y * height)),
                                    (int(landmarks[i+1].x * width), int(landmarks[i+1].y * height)),
                                    160, 10  # Gray with thickness 10
                                )
                        
                        for i in range(12, 18):  # Right arm
                            if i < len(landmarks)-1:
                                cv2.line(
                                    parse_array,
                                    (int(landmarks[i].x * width), int(landmarks[i].y * height)),
                                    (int(landmarks[i+1].x * width), int(landmarks[i+1].y * height)),
                                    160, 10  # Gray with thickness 10
                                )
                        
                        # Draw legs
                        for i in range(23, 29):  # Left leg
                            if i < len(landmarks)-1:
                                cv2.line(
                                    parse_array,
                                    (int(landmarks[i].x * width), int(landmarks[i].y * height)),
                                    (int(landmarks[i+1].x * width), int(landmarks[i+1].y * height)),
                                    200, 15  # Light gray with thickness 15
                                )
                        
                        for i in range(24, 30):  # Right leg
                            if i < len(landmarks)-1:
                                cv2.line(
                                    parse_array,
                                    (int(landmarks[i].x * width), int(landmarks[i].y * height)),
                                    (int(landmarks[i+1].x * width), int(landmarks[i+1].y * height)),
                                    200, 15  # Light gray with thickness 15
                                )
                        
                        # Convert back to PIL and save
                        parse_img = Image.fromarray(parse_array)
                    
                    parse_img.save(parse_file)
                except Exception as e:
                    print(f"Error generating parse map for {file_path}: {str(e)}")
                
                # Close images
                img.close()
                crop.close()
                mask.close()
                
            except Exception as e:
                print(f"Error processing {dest_img}: {str(e)}")
                
        except Exception as e:
            print(f"Error with record {file_path}: {str(e)}")
            
    pose_model.close()
    return found_count

def main():
    print("DeepFashion Dataset Preparation")
    print("=" * 40)
    
    # Check paths
    if not Path(CSV_FILE).exists() or not Path(SOURCE_IMAGES).exists():
        print(f"Error: Input paths not found!")
        print(f"CSV: {Path(CSV_FILE).exists()}, Images: {Path(SOURCE_IMAGES).exists()}")
        return
    
    # Create directories
    for split in ['train', 'test']:
        for folder in ['image', 'pose', 'image-parse', 'cloth', 'cloth-mask']:
            Path(OUTPUT_DIR, split, folder).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading CSV data...")
    df = pd.read_csv(CSV_FILE)
    df = df[df['split'].isin(['train', 'test'])].copy()
    df = df.dropna(subset=['file', 'split', 'x_1', 'y_1', 'x_2', 'y_2'])
    
    if MAX_SAMPLES:
        df = df.sample(min(MAX_SAMPLES, len(df)))
        print(f"Using {len(df)} sample records")
    else:
        print(f"Processing {len(df)} records")
    
    # Test image paths
    print("\nTesting first few image paths...")
    for i in range(min(3, len(df))):
        file_path = df.iloc[i]['file']
        img_path = find_image(SOURCE_IMAGES, file_path)
        print(f"{'✅ Found' if img_path else '❌ Not found'}: {file_path}")
        if img_path:
            print(f"  → {img_path}")
    
    # Process in batches
    batch_size = 500
    total_found = 0
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size+1}/{(len(df)+batch_size-1)//batch_size}")
        found = process_batch(batch)
        total_found += found
    
    print(f"\nProcessed {total_found}/{len(df)} images successfully")
    
    # Print statistics
    print("\nDataset Statistics:")
    base_path = Path(OUTPUT_DIR)
    for split in ['train', 'test']:
        print(f"\n{split.upper()}:")
        for folder in ['image', 'pose', 'cloth', 'cloth-mask', 'image-parse']:
            folder_path = base_path / split / folder
            if folder_path.exists():
                count = len(list(folder_path.glob('**/*.*')))
                print(f"  {folder}: {count} files")

if __name__ == '__main__':
    main()