# Data Preparation Plan

## 1. Data Collection

### Fashion Dataset

- **Download Datasets:**
  - DeepFashion Dataset
  - Fashion-Gen Dataset
- **Contents:**
  - Images and descriptions of fashion items.
- **Requirements:**
  - Ensure datasets include annotations for clothing attributes, categories, and bounding boxes where applicable.

### Pose Estimation Data

- **Download or Setup:**
  - Obtain OpenPose model data or use MediaPipe for body pose extraction.
- **Tasks:**
  - Set up pose estimation tool.
  - Gather sample images with pose annotations.

---

## 2. Data Inspection

- **Fashion Datasets:**
  - Load datasets and inspect images, labels, and metadata (e.g., attributes like color, sleeve length, clothing type).
- **Pose Estimation Data:**
  - Verify that pose detection correctly annotates body poses.

---

## 3. Data Cleaning

### Fashion Dataset

- Remove corrupted or incomplete images.
- Filter out images without proper clothing descriptions or annotations.
- Normalize image dimensions (e.g., resize to **256x256 pixels**).

### Pose Data

- Ensure body pose data is accurate for all sample images.
- Remove outliers where pose detection failed.

---

## 4. Data Labeling

### Fashion Dataset

- Organize and label images into categories such as **tops, bottoms, dresses, outerwear**, etc.
- Verify all images have correct labels and associated metadata.

### Pose Estimation

- Generate pose annotations (keypoints for arms, legs, torso, etc.) for each fashion image.
- Save pose data alongside images in a required format (e.g., **JSON** or **CSV**).

---

## 5. Data Augmentation

- **Techniques:**
  - Random cropping
  - Horizontal flipping
  - Adjusting brightness and contrast
  - Adding noise or slight rotation
- **Note:** Ensure augmented data retains labels and pose annotations.

---

## 6. Documentation

- Document preprocessing steps:
  - Methods used for cleaning and labeling data.
  - Augmentation techniques applied.
- Maintain a clear and organized **folder structure** (e.g., sorted by category).