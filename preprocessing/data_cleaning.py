"""
we will do the following:
- checks for corrupted images
- resizes and pads to square images 
- saves into a structured cleaned folder 
- use parallel processing with ThreadPoolExecutor to accelerate the processing
"""

# ----------------------------------------------------------------------------------- #
# TODO 1 - import necessary libraries
from pathlib import Path
from PIL import Image
import cv2
import albumentations as A
from tqdm.auto import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocessing import data_loader as dl
from preprocessing import config as C
import json
import shutil
import os

# ----------------------------------------------------------------------------------- #
# TODO 2 - Check if an image is corrupted
def is_corrupted(src_image: Path):
    """
    - verify that the image is not corrupted or not incomplete
    - check that the file header is correct
    - img.verify() does not load the image pixels into memory
    
    Returns True if image cannot be opened or verified
    """
    try:
        if not src_image.exists():
            return True
        with Image.open(src_image) as img:
            img.verify()
        return False
    except Exception as e:
        print(f"Corruption check failed for {src_image}: {e}")
        return True

# ----------------------------------------------------------------------------------- #
# TODO 3 - Albumentations resize + pad
pad_resize = A.Compose([
    A.LongestMaxSize(
        max_size=C.IMG_SIZE,
        interpolation=cv2.INTER_CUBIC
    ),
    A.PadIfNeeded(
        C.IMG_SIZE,
        C.IMG_SIZE,
        border_mode=cv2.BORDER_CONSTANT
    )
])

# ----------------------------------------------------------------------------------- #
# TODO 4 - Resize and save a single image
def resize_and_save(src_file: Path, dst: Path):
    """
    - Reads image
    - augment image
    - save to dst path
    """
    try:
        # Check if source file exists
        if not src_file.exists():
            print(f"Source file does not exist: {src_file}")
            return False
            
        img = cv2.imread(str(src_file))
        if img is None:
            print(f"Could not read image: {src_file}")
            return False
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pad_resize(image=img)["image"]
        
        # Create destination directory
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        success = cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            print(f"Failed to save image: {dst}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error during resize and save function: {str(e)}")
        return False

# ----------------------------------------------------------------------------------- #
# TODO 5 - Process a single image
def process_image(row, root_path: Path, data_name: str):
    """
    parameters:
        - row: an iterable form the df annotation that we create from data_loader script
        - root_path: path to source image
        - data_name: DeepFashion or FashionGen dataset
    goals:
        - Check if the image is corrupted:
            - ignore it
        - else:
            - resize and save that image into a clean path
    """
    try:
        src = Path(root_path) / row.file
        clean_root = C.CLEAN_DIR / data_name

        # Check if source file exists
        if not src.exists():
            print(f"Source file does not exist: {src}")
            return False
        
        # check that the image format is correct
        exts = ['.png', '.jpeg', '.jpg', '.JPEG', '.JPG', '.PNG']
        if src.suffix.lower() not in [ext.lower() for ext in exts]:
            print(f"Unsupported format skipped: {src}")
            return False
        
        # if corrupted return False
        if is_corrupted(src):
            print(f"Corrupted image skipped: {src}")
            return False

        # build the destination folder to copy the image in the dest folder
        dst = clean_root / "Images" / row.file
        
        # Try to resize and save
        success = resize_and_save(src, dst)
        if not success:
            print(f"Failed to process {str(src)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return False

# ----------------------------------------------------------------------------------- #
# TODO 6 - Clean folder using threads for fast processing
def clean_folder(df: pd.DataFrame, root_path: Path, data_name: str, max_workers: int = 12):
    """
    Parameters:
        - df: our annotation df
        - root_path: base path to original images 
        - data_name: either DeepFashion or FashionGen
        - max_workers: number of threads to run in parallel
    """
    print(f"Starting cleaning process for {data_name}")
    print(f"Total records in CSV: {len(df)}")
    print(f"Root path: {root_path}")
    
    # Check if root path exists
    if not Path(root_path).exists():
        print(f"ERROR: Root path does not exist: {root_path}")
        return
    
    df = df.copy()
    rows = df.itertuples(index=False)
    rows = list(rows)
    
    successful_count = 0
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, row, root_path, data_name)
            for row in rows
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="cleaning/resizing/saving_to_clean_folder"):
            try:
                result = future.result()
                if result:
                    successful_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"Future execution error: {e}")
                skipped_count += 1
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total images in CSV: {len(rows)}")
    print(f"Total images successfully cleaned: {successful_count}")
    print(f"Total images skipped/failed: {skipped_count}")

if __name__ == "__main__":
    deep_data_path = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Dataset_Clean\DeepFashion\annotation\DeepFashion.csv"
    gen_data_path = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Dataset_Clean\FashionGen\annotation\FashionGen.csv"
    
    deepFashion = pd.read_csv(deep_data_path)
    Fashion_gen = pd.read_csv(gen_data_path)
    
    root_path_deep_fashion = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Data\DeepFashion\Images"
    root_path_fashion_gen = r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Data\FashionGen\images"
    
    # # Process both datasets
    # print("=" * 60)
    # print("PROCESSING FASHIONGEN DATASET")
    # print("=" * 60)
    # clean_folder(Fashion_gen, root_path_fashion_gen, "FashionGen", 12)
    
    print("\n" + "=" * 60)
    print("PROCESSING DEEPFASHION DATASET")
    print("=" * 60)
    clean_folder(deepFashion, root_path_deep_fashion, "DeepFashion", 12)