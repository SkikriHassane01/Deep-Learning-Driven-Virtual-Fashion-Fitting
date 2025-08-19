import os
from pathlib import Path
import json
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# Import for profiling
try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


# -------------------------
#   U-Net style generator
# -------------------------
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        x = torch.cat((x, skip), dim=1)
        return x


class TryOnGenerator(nn.Module):
    """
    Inputs (C=9):
      person (3) + cloth (3) + cloth_mask (1) + pose (1) + body_mask(1)
    Outputs (C=3): synthesized image
    """
    def __init__(self, in_channels=9, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)


class VirtualTryOnModel:
    """
    Wrapper class for TryOnGenerator to be used in production/inference.
    Provides simplified loading and inference methods for Django application.
    """
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TryOnGenerator(in_channels=9, out_channels=3)
        self.model = self.model.to(self.device)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Default transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded model weights from: {checkpoint_path}")
    
    def predict(self, person_img, cloth_img, cloth_mask=None, pose_map=None, body_mask=None):
        """
        Run inference with the model
        
        Args:
            person_img: PIL Image of person
            cloth_img: PIL Image of clothing item
            cloth_mask: Optional PIL Image of clothing mask (will generate if None)
            pose_map: Optional PIL Image of pose keypoints (will use empty if None)
            body_mask: Optional PIL Image of body segmentation (will use full mask if None)
            
        Returns:
            PIL Image of the virtual try-on result
        """
        # Prepare input
        if cloth_mask is None:
            cloth_mask = Image.new("L", cloth_img.size, 255)  # Default white mask
        
        if pose_map is None:
            pose_map = Image.new("L", person_img.size, 0)  # Empty pose map
            
        if body_mask is None:
            body_mask = Image.new("L", person_img.size, 255)  # Full body mask
        
        # Apply transforms
        person_tensor = self.transform(person_img)
        cloth_tensor = self.transform(cloth_img)
        cloth_mask_tensor = self.mask_transform(cloth_mask)
        pose_map_tensor = self.mask_transform(pose_map)
        body_mask_tensor = self.mask_transform(body_mask)
        
        # Create input tensor
        x = torch.cat([person_tensor, cloth_tensor, cloth_mask_tensor, 
                      pose_map_tensor, body_mask_tensor], dim=0)
        x = x.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            output = self.model(x)
        
        # Convert to image
        output_tensor = output.squeeze().cpu()
        output_img = self.tensor_to_pil(output_tensor)
        
        return output_img
    
    @staticmethod
    def tensor_to_pil(tensor):
        """Convert a normalized tensor to PIL Image"""
        # Denormalize
        tensor = (tensor * 0.5) + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(array)


# -------------------------
#   Dataset (matching your structure)
# -------------------------
class VirtualTryOnDataset(Dataset):
    """
    Expects:
      data/
        train/ or test/
          image/           (person images)
          cloth/           (cloth images)
          cloth-mask/      (masks; usually PNG)
          openpose_json/   (person_keypoints.json)
      data/train_pairs.txt (person.jpg cloth.jpg)
      data/test_pairs.txt
    """
    def __init__(self, data_dir, pairs_file, transform=None, pose_transform=None, 
                 split='train', cache_images=False):
        self.data_dir = Path(data_dir)
        self.split = split  # "train" or "test"
        self.transform = transform
        self.pose_transform = pose_transform
        self.cache_images = cache_images
        self.cached_data = {}

        pairs_path = self.data_dir / pairs_file
        self.pairs = []
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p, c = line.split()
                self.pairs.append((p, c))

        self.split_dir = self.data_dir / self.split
        self.person_dir = self.split_dir / "image"
        self.cloth_dir  = self.split_dir / "cloth"
        self.mask_dir   = self.split_dir / "cloth-mask"
        self.pose_dir   = self.split_dir / "openpose_json"
        
        # Pre-cache all the files if requested
        if self.cache_images:
            print(f"Caching {len(self.pairs)} {split} images in memory...")
            for idx in tqdm(range(len(self.pairs))):
                self._cache_item(idx)

    def _cache_item(self, idx):
        """Pre-load and cache an item"""
        if idx in self.cached_data:
            return
            
        person_name, cloth_name = self.pairs[idx]

        # paths
        person_path = self.person_dir / person_name
        cloth_path  = self.cloth_dir / cloth_name
        mask_path   = self._find_mask(self.mask_dir, cloth_name)
        pose_json = self.pose_dir / (Path(person_name).stem + "_keypoints.json")

        # load images (RGB)
        try:
            person_img = Image.open(person_path).convert("RGB")
            cloth_img  = Image.open(cloth_path).convert("RGB")

            # cloth mask (L)
            if mask_path and mask_path.exists():
                cloth_mask = Image.open(mask_path).convert("L")
            else:
                cloth_mask = Image.new("L", cloth_img.size, 255)

            # pose map (L)
            pose_map = (
                self._load_pose_map(pose_json, size_hw=(person_img.size[1], person_img.size[0]))
                if pose_json.exists() else Image.new("L", person_img.size, 0)
            )

            # body mask (placeholder = all 1s)
            body_mask = Image.new("L", person_img.size, 255)
            
            self.cached_data[idx] = (person_img, cloth_img, cloth_mask, pose_map, body_mask)
        except Exception as e:
            print(f"Warning: Failed to cache item {idx}: {e}")

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _find_mask(mask_dir: Path, cloth_name: str) -> Path | None:
        """
        Try common mask filenames:
          - same name (jpg/png)
          - <stem>.png
          - <stem>_mask.png
        """
        stem = Path(cloth_name).stem
        candidates = [
            mask_dir / cloth_name,                  # same filename
            mask_dir / f"{stem}.png",               # stem.png
            mask_dir / f"{stem}_mask.png",          # stem_mask.png
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_pose_map(self, json_path: Path, size_hw: tuple[int, int]) -> Image.Image:
        """
        Draw keypoints from OpenPose JSON to a single-channel pose map.
        size_hw = (H, W)
        """
        H, W = size_hw
        pose_map = np.zeros((H, W), dtype=np.uint8)
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # OpenPose format: people[0].pose_keypoints_2d = [x1,y1,conf1,...]
            keypoints = []
            if isinstance(data, dict):
                if "people" in data and data["people"]:
                    keypoints = data["people"][0].get("pose_keypoints_2d", [])
                elif "pose_keypoints_2d" in data:
                    keypoints = data["pose_keypoints_2d"]
                elif "keypoints" in data:
                    keypoints = data["keypoints"]

            for i in range(0, len(keypoints), 3):
                if i + 2 >= len(keypoints):
                    break
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                if conf is None or conf < 0.1:
                    continue
                xi = int(round(x))
                yi = int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    cv2.circle(pose_map, (xi, yi), 3, 255, -1)
        except Exception as e:
            # If anything fails, leave it empty
            print(f"[warn] pose parse failed for {json_path.name}: {e}")

        return Image.fromarray(pose_map, mode="L")

    def __getitem__(self, idx):
        if self.cache_images and idx in self.cached_data:
            # Use cached data
            person_img, cloth_img, cloth_mask, pose_map, body_mask = self.cached_data[idx]
        else:
            # Load from disk
            person_name, cloth_name = self.pairs[idx]

            # paths
            person_path = self.person_dir / person_name
            cloth_path  = self.cloth_dir / cloth_name
            mask_path   = self._find_mask(self.mask_dir, cloth_name)
            pose_json = self.pose_dir / (Path(person_name).stem + "_keypoints.json")

            # load images (RGB)
            if not person_path.exists():
                raise FileNotFoundError(f"Person not found: {person_path}")
            if not cloth_path.exists():
                raise FileNotFoundError(f"Cloth not found: {cloth_path}")

            person_img = Image.open(person_path).convert("RGB")
            cloth_img  = Image.open(cloth_path).convert("RGB")

            # cloth mask (L)
            if mask_path and mask_path.exists():
                cloth_mask = Image.open(mask_path).convert("L")
            else:
                cloth_mask = Image.new("L", cloth_img.size, 255)

            # pose map (L)
            pose_map = (
                self._load_pose_map(pose_json, size_hw=(person_img.size[1], person_img.size[0]))
                if pose_json.exists() else Image.new("L", person_img.size, 0)
            )

            # body mask (placeholder = all 1s)
            body_mask = Image.new("L", person_img.size, 255)

        # transforms (resize to 512x512)
        if self.transform is not None:
            person_img = self.transform(person_img)
            cloth_img  = self.transform(cloth_img)
        if self.pose_transform is not None:
            cloth_mask = self.pose_transform(cloth_mask)
            pose_map   = self.pose_transform(pose_map)
            body_mask  = self.pose_transform(body_mask)

        # concat channels: 3 + 3 + 1 + 1 + 1 = 9
        inp = torch.cat([person_img, cloth_img, cloth_mask, pose_map, body_mask], dim=0)
        target = person_img  # naive target (identity); for real try-on you'd use ground-truth pairs
        return inp, target


# Custom learning rate scheduler with warmup
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_lr = self.base_lrs.copy()
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = epoch / max(1, self.warmup_epochs)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale
                self.last_lr[i] = param_group['lr']
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.eta_min + (self.base_lrs[i] - self.eta_min) * cosine_factor
                self.last_lr[i] = param_group['lr']
    
    def get_last_lr(self):
        return self.last_lr


def train_model(data_dir, output_dir, epochs=50, batch_size=4, lr=0.0002, resume_checkpoint=None,
               num_workers=4, cache_data=False, benchmark=False, save_every=10):
    """
    Train the virtual try-on model
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to save checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        resume_checkpoint: Path to resume training from (optional)
        num_workers: Number of data loading workers
        cache_data: Whether to cache dataset in memory
        benchmark: Set cudnn.benchmark for potentially faster training
        save_every: Save checkpoint every N epochs
    
    Returns:
        Path to best model weights
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDNN benchmark enabled")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    pose_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # Create datasets and dataloaders
    train_dataset = VirtualTryOnDataset(
        data_dir=data_dir,
        pairs_file="train_pairs.txt",
        transform=transform,
        pose_transform=pose_transform,
        split="train",
        cache_images=cache_data
    )
    
    test_dataset = VirtualTryOnDataset(
        data_dir=data_dir,
        pairs_file="test_pairs.txt",
        transform=transform,
        pose_transform=pose_transform,
        split="test",
        cache_images=cache_data
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create model
    model = TryOnGenerator(in_channels=9, out_channels=3)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Learning rate scheduler
    sched = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        max_epochs=epochs,
        eta_min=1e-6
    )
    
    # Loss functions
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best = float('inf')
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'best' in checkpoint:
                    best = checkpoint['best']
                print(f"Resumed from checkpoint: {resume_checkpoint} (epoch {start_epoch})")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights from: {resume_checkpoint}")
        except RuntimeError as e:
            if "WeightsUnpickler error" in str(e):
                print("Warning: Checkpoint loading failed with weights_only=True")
                print("Attempting to load with weights_only=False (less secure but more compatible)")
                try:
                    # Try with weights_only=False (less secure but compatible with older PyTorch)
                    checkpoint = torch.load(
                        resume_checkpoint, 
                        map_location=device, 
                        weights_only=False
                    )
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                        if 'optimizer' in checkpoint:
                            optimizer.load_state_dict(checkpoint['optimizer'])
                        if 'epoch' in checkpoint:
                            start_epoch = checkpoint['epoch'] + 1
                        if 'best' in checkpoint:
                            best = checkpoint['best']
                        print(f"Resumed from checkpoint: {resume_checkpoint} (epoch {start_epoch})")
                    else:
                        model.load_state_dict(checkpoint)
                        print(f"Loaded model weights from: {resume_checkpoint}")
                except Exception as e2:
                    # If both approaches fail, raise the original error
                    print(f"Error loading checkpoint: {e}")
                    raise e
            else:
                # For other types of errors, raise them directly
                raise e
    
    # Initialize AMP gradient scaler
    scaler = GradScaler()
    
    # Path for best model weights
    best_weights = output_dir / "best_model.pth"
    
    print(f"Starting training for {epochs} epochs...")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        # Update scheduler at the beginning of each epoch
        sched.step(epoch)
        
        model.train()
        tr_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Use AMP for mixed precision training
            with autocast():
                y_hat = model(x)
                loss = l1(y_hat, y) + 0.1 * mse(y_hat, y)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tr_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Validation
        model.eval()
        te_loss = 0.0
        
        pbar = tqdm(test_loader, desc="Validating")
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = l1(y_hat, y) + 0.1 * mse(y_hat, y)
                te_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        tr_loss /= max(1, len(train_loader))
        te_loss /= max(1, len(test_loader))
        sched.step(epoch)
        
        print(f"Epoch {epoch+1}: train={tr_loss:.4f}  test={te_loss:.4f}  lr={sched.get_last_lr()[0]:.6f}")
        
        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best': best,
                'scaler': scaler.state_dict()  # Save scaler state
            }, checkpoint_path)
        
        # Save best weights
        if te_loss < best:
            best = te_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best': best,
                'scaler': scaler.state_dict()  # Save scaler state
            }, best_weights)
            print(f"  ↳ saved new best to {best_weights} (test {best:.4f})")
    
    print("Training completed!")
    print(f"Best test loss: {best:.4f}")
    return best_weights
