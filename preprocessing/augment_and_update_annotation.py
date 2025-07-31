import pandas as pd
from pathlib import Path
from preprocessing.data_augmentation import Augment
from PIL import Image
import os
from tqdm import tqdm
ANNOTATION_CSV = Path(r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Dataset_Clean\DeepFashion\annotation\DeepFashion.csv")
IMAGE_ROOT = Path(r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Dataset_Clean\DeepFashion\Images")
AUG_SUFFIX = "_aug"

def augment_train_images_and_update_annotations():
    df = pd.read_csv(ANNOTATION_CSV)
    augmenter = Augment()

    new_rows = []
    
    train_rows = df[df['split'] == 'train']
    for _, row in tqdm(train_rows.iterrows(), total=len(train_rows), desc="Augmenting train images"):
        orig_rel_path = row['file']
        # Ensure orig_rel_path is a Path, not a string with backslashes
        orig_rel_path = Path(orig_rel_path)
        orig_abs_path = IMAGE_ROOT / orig_rel_path

        if not orig_abs_path.exists():
            print(f"Image not found: {orig_abs_path}")
            continue
        
        with Image.open(orig_abs_path).convert("RGB") as img:
            augmented = augmenter(img)

            # Save augmented image next to original with suffix
            aug_filename = orig_rel_path.stem + AUG_SUFFIX + ".jpg"
            aug_rel_path = orig_rel_path.parent / aug_filename
            aug_abs_path = IMAGE_ROOT / aug_rel_path
            aug_abs_path.parent.mkdir(parents=True, exist_ok=True)
            augmented.save(aug_abs_path)
            
            new_row = row.copy()
            new_row['file'] = str(aug_rel_path).replace("\\", "/")
            new_rows.append(new_row)
            
    # Append new rows to original dataframe and save
    df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_aug.to_csv("DeepFashion_augmented.csv", index=False)
    print(f"✅ Augmented images saved and annotation updated at DeepFashion_augmented.csv")
    
if __name__ == "__main__":
    augment_train_images_and_update_annotations()