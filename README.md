# Virtual Fashion Try-On

A deep learning-based virtual try-on system that allows users to see how clothes would look on them.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Automatic Setup and Training

```bash
# Run the complete setup and training pipeline
python setup_and_train.py

# Or with custom parameters
python setup_and_train.py --epochs 100 --batch-size 8 --lr 0.0001
```

### 3. Manual Step-by-Step Setup

#### Step 1: Download Models
```bash
python download_models.py
```

#### Step 2: Prepare Your Dataset
1. Place your images in the following structure:
```
data/
├── person/          # Person images
├── cloth/           # Clothing images  
├── cloth-mask/      # Clothing masks (optional)
├── pose/            # Pose keypoints (optional)
├── train_pairs.txt  # Training pairs
└── test_pairs.txt   # Test pairs
```

2. Create training pairs file (`data/train_pairs.txt`):
```
person_001.jpg cloth_001.jpg
person_002.jpg cloth_002.jpg
...
```

#### Step 3: Validate Data
```bash
python train.py --validate-only --data-dir ./data
```

#### Step 4: Train Model
```bash
python train.py --data-dir ./data --epochs 50 --batch-size 4
```

#### Step 5: Evaluate Model
```bash
python evaluate.py --model-path ./checkpoints/best_model_weights.pth --test-dir ./data
```

## Dataset Format

### Required Files
- `train_pairs.txt`: Training pairs in format "person_image cloth_image"
- `test_pairs.txt`: Test pairs in format "person_image cloth_image"

### Required Directories
- `person/`: Person images
- `cloth/`: Clothing images

### Optional Directories
- `cloth-mask/`: Clothing masks (auto-generated if missing)
- `pose/`: Pose keypoint JSON files (auto-generated if missing)

## Training Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 4)
- `--lr`: Learning rate (default: 0.0002)
- `--resume`: Resume from checkpoint
- `--data-dir`: Path to dataset directory
- `--output-dir`: Path to save checkpoints

## Model Architecture

The system uses a U-Net based generator that takes:
- Person image (3 channels)
- Cloth image (3 channels)
- Cloth mask (1 channel)
- Pose map (1 channel)
- Body mask (1 channel)

Total input: 9 channels → 3 channel output (try-on result)

## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model download fails**: Check internet connection and try again
4. **Training data not found**: Verify data directory structure

### GPU Requirements

- Recommended: NVIDIA GPU with 8GB+ VRAM
- Minimum: 4GB VRAM (reduce batch size to 1-2)
- CPU training is supported but much slower

## File Structure

```
VirtualFashion/
├── models/
│   ├── pose/              # OpenPose models
│   ├── segmentation/      # U2Net models
│   ├── tryon_model.py     # Main try-on model
│   └── u2net.py          # U2Net implementation
├── data/                  # Dataset directory
├── checkpoints/           # Model checkpoints
├── evaluation_results/    # Evaluation outputs
├── requirements.txt       # Python dependencies
├── setup_and_train.py    # Complete setup script
├── download_models.py    # Model download script
├── train.py              # Training script
├── evaluate.py           # Evaluation script
└── README.md            # This file
```
