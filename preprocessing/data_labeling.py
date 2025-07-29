import os

# Suppress MediaPipe and TensorFlow logs to avoid warnings
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs

import pandas as pd 
from pathlib import Path
from preprocessing import config as C
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed  # Changed from ProcessPoolExecutor
import mediapipe as mp
import cv2
import json
import threading

# Thread-local storage for keeping a separate pose model per thread
thread_local = threading.local()

def get_pose_model():
    """
    Initialize the MediaPipe Pose model only once per thread.
    Avoids reloading the model for every single image.
    """
    if not hasattr(thread_local, "pose_model"):
        thread_local.pose_model = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    return thread_local.pose_model


def categories_to_ids(df: pd.DataFrame, data_path: Path):
    """
    Convert the 'category' column to integer label IDs
    and update the CSV with the new 'label_id' column.
    """
    categories = sorted(df['category'].unique())
    cat2id = {
        cat: idx for idx, cat in enumerate(categories)
    }

    df["label_id"] = df["category"].map(cat2id)
    df.to_csv(data_path, index=False)


def _process_single_image(img_path_str, out_root_str, relative_file_path):
    """
    Function that:
        - loads images from the cleaned folder structure
        - runs MediaPipe pose detection
        - saves keypoints as JSON in the same flat structure
    
    Args:
        img_path_str: path to the image file
        out_root_str: output root directory path
        relative_file_path: relative path from annotation (row.file)
    """
    img_path = Path(img_path_str)
    out_root = Path(out_root_str)
    
    # Check if image file exists
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return False
    
    relative_path = Path(relative_file_path)
    out_dir = out_root / relative_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load image: {img_path}")
        return False

    # Optional: Resize image to reduce processing time
    img = cv2.resize(img, (256, 256))

    try:
        # Get thread-local pose model
        pose = get_pose_model()

        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_img)
        
        # Gather keypoints
        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    "x": round(landmark.x, 5),
                    "y": round(landmark.y, 5),
                    "z": round(landmark.z, 5),
                    "visibility": round(landmark.visibility, 5)
                })
        
        # Create output filename - same as image but with _pose.json extension
        out_filename = f"{relative_path.stem}_pose.json"
        out_path = out_dir / out_filename
        
        # Save keypoints to JSON
        pose_data = {
            "file": relative_file_path,
            "landmarks": keypoints,
            "num_landmarks": len(keypoints)
        }
        
        with open(out_path, "w") as f:
            json.dump(pose_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def label_all_poses_parallel(data_name="DeepFashion", max_workers=None):
    """
    Process pose detection for all images in the cleaned dataset
    Saves pose JSON files in flat structure: POSE_DIR/data_name/
    
    Args:
        data_name: "DeepFashion" or "FashionGen"
        max_workers: number of parallel threads
    """
    # Paths
    base = C.CLEAN_DIR / data_name
    out_root = C.POSE_DIR / data_name
    annotation_file = base / "annotation" / f"{data_name}.csv"
    
    print(f"Processing poses for {data_name}")
    print(f"Base directory: {base}")
    print(f"Output directory: {out_root}")
    print(f"Annotation file: {annotation_file}")
    
    # Check if annotation file exists
    if not annotation_file.exists():
        print(f"ERROR: Annotation file not found: {annotation_file}")
        print("Make sure you have run the cleaning script first!")
        return
    
    # Load the cleaned CSV
    try:
        df = pd.read_csv(annotation_file)
        print(f"Loaded {len(df)} records from annotation file")
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return
    
    img_info_list = []
    missing_files = 0
    
    for _, row in df.iterrows():
        # Construct full image path
        img_path = base / "Images" / row['file'] 
        
        if img_path.exists():
            img_info_list.append((
                str(img_path),
                str(out_root),
                row['file']  # This is the relative path from annotation
            ))
        else:
            missing_files += 1
            print(f"Missing file: {img_path}")
    
    print(f"Found {len(img_info_list)} valid images")
    if missing_files > 0:
        print(f"WARNING: {missing_files} files from CSV not found on disk")
    
    if not img_info_list:
        print("No valid images found to process!")
        return
    
    successful = 0
    failed = 0

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(_process_single_image, *img_info)
            for img_info in img_info_list
        ]
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), 
                          total=len(futures), 
                          desc=f"Processing poses for {data_name}"):
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Future execution error: {e}")
                failed += 1
    
    print(f"\n=== POSE PROCESSING SUMMARY ===")
    print(f"Total images processed: {len(img_info_list)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {out_root}")


if __name__ == "__main__":
    # Load annotation files for both datasets
    deep_fashion_annotation = C.CLEAN_DIR / "DeepFashion" / "annotation" / "deepFashion.csv"
    fashion_gen_annotation = C.CLEAN_DIR / "FashionGen" / "annotation" / "FashionGen.csv"

    deep_fashion_df = pd.read_csv(deep_fashion_annotation)
    fashion_gen_df = pd.read_csv(fashion_gen_annotation)

    # Uncomment below to process category to integer IDs
    print("=" * 60)
    print("PROCESSING CATEGORY MAPPING")
    print("=" * 60)
    categories_to_ids(deep_fashion_df, deep_fashion_annotation)
    categories_to_ids(fashion_gen_df, fashion_gen_annotation)
    
    # Start pose detection
    print("\n" + "=" * 60)
    print("PROCESSING POSE DETECTION")
    print("=" * 60)
    
    print("\nStarting FashionGen pose detection...")
    label_all_poses_parallel("FashionGen", max_workers=12)

    # Process pose detection
    print("Starting DeepFashion pose detection...")
    label_all_poses_parallel("DeepFashion", max_workers=12)