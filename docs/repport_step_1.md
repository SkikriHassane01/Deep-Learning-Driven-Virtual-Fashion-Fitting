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
│           └── FashionGen.csv
├── Pose/
│   ├── DeepFashion/          # JSON keypoints
│   └── FashionGen/           # JSON keypoints
└── preprocessing/
    ├── data_cleaning.py
    ├── data_labeling.py
    ├── data_augmentation.py
    ├── augment_and_update_annotation.py
    └── config.py
```