# Data Preprocessing Summary

A quick overview of how we prepare our fashion datasets before training.

---

## Folder Layout After Processing

```
Data/
├── DeepFashion/              # Raw images
├── FashionGen/               # Raw images
├── Dataset_Clean/
│   ├── DeepFashion/
│   │   ├── Images/           # Cleaned & resized
│   │   └── annotation/
│   │       ├── DeepFashion.csv
│   │       └── DeepFashion_augmented.csv
│   └── FashionGen/
│       ├── Images/           # Cleaned & resized
│       └── annotation/
├── Pose/
│   ├── DeepFashion/          # JSON keypoints
│   └── FashionGen/           # JSON keypoints
└── preprocessing/
```


---

## 1. Cleaning Images

- **What:** Remove broken files, resize to 256×256, pad to square.
- **How:**  
  1. Verify each image can open.  
  2. Use Albumentations to resize & pad.  
  3. Run in parallel threads for speed.  
- **Result:** Clean images in `Dataset_Clean/[Dataset]/Images/`.

---

## 2. Labeling Poses

- **What:** Detect 33 body landmarks per image.
- **How:**  
  1. Run MediaPipe Pose on cleaned images.  
  2. Save coordinates + visibility in `[name]_pose.json`.  
  3. Filter out bad JSON files (missing points or low visibility).  
- **Result:** JSON files in `Pose/[Dataset]/`.

---

## 3. Augmenting Data

- **What:** Create variations of training images to boost model robustness.
- **Techniques:**  
  - Random crop & resize  
  - Horizontal flip  
  - Brightness/contrast change  
  - Small rotations  
  - Gaussian noise  
- **How:**  
  1. Wrap these transforms in an `Augment` class.  
  2. Apply only to training set.  
  3. Save new images with a `_aug` suffix.  
  4. Update `DeepFashion_augmented.csv`.  
- **Result:** More diverse training data.

---

## 4. Annotations

- **What:** Combine file paths, categories, bounding boxes, and split info.
- **How:**  
  1. Load raw metadata files.  
  2. Merge into a single CSV per dataset.  
- **Result:**  
  - `DeepFashion.csv`  
  - `FashionGen.csv`  

---