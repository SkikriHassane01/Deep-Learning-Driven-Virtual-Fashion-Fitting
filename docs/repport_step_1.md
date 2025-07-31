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