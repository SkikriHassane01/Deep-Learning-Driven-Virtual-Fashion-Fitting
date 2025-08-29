from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import json
import os
import cv2
import torch.serialization as ts
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageDraw
from tqdm import tqdm


# -------------------------
#   Safe checkpoint loader (PyTorch 2.6+)
# -------------------------
def load_checkpoint_trusted(path, map_location):
    """
    Load older checkpoints created before PyTorch 2.6.
    Uses weights_only=False and allowlists numpy's scalar class.
    ONLY use this if you trust the checkpoint source.
    """
    ts.add_safe_globals([np.core.multiarray.scalar])
    return torch.load(path, map_location=map_location, weights_only=False)


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
      person(3) + cloth(3) + cloth_mask(1) + pose(1) + body_mask(1)
    Output (C=3)
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
            nn.Tanh(),
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


# -------------------------
#   Dataset (matches your structure)
# -------------------------
class VirtualTryOnDataset(Dataset):
    """
    data/
      train/ or test/
        image/           (person images)
        cloth/           (cloth images)
        cloth-mask/      (masks)
        openpose_json/   (pose JSON)
    data/train_pairs.txt (person.jpg cloth.jpg)
    data/test_pairs.txt
    """
    def __init__(self, data_dir, pairs_file, transform=None, pose_transform=None, split="train", cache_images=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.pose_transform = pose_transform
        self.cache_images = cache_images
        self.cached_data = {}

        # Load pairs file - fixed duplicate parsing
        pairs_path = self.data_dir / pairs_file
        self.pairs = []
        try:
            with pairs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        self.pairs.append((parts[0], parts[1]))
        except FileNotFoundError:
            print(f"Warning: Pairs file {pairs_path} not found. Creating empty dataset.")
            self.pairs = []

        self.split_dir = self.data_dir / self.split
        self.person_dir = self.split_dir / "image"
        self.cloth_dir = self.split_dir / "cloth"
        self.mask_dir = self.split_dir / "cloth-mask"
        self.pose_dir = self.split_dir / "openpose_json"

        # Cache images if requested - fixed duplicate caching
        if self.cache_images and len(self.pairs) > 0:
            print(f"Caching {len(self.pairs)} {split} items in memory...")
            for idx in tqdm(range(len(self.pairs))):
                self._cache_item(idx)

    @staticmethod
    def _find_mask(mask_dir: Path, cloth_name: str):
        """Find mask file for given cloth name"""
        stem = Path(cloth_name).stem
        candidates = [
            mask_dir / cloth_name,
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}_mask.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_pose_map(self, json_path: Path, size_hw):
        """Load pose map from JSON file"""
        H, W = size_hw
        pose_map = np.zeros((H, W), dtype=np.uint8)
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            keypoints = []
            if isinstance(data, dict):
                if "people" in data and data["people"]:
                    keypoints = data["people"][0].get("pose_keypoints_2d", [])
                elif "pose_keypoints_2d" in data:
                    keypoints = data["pose_keypoints_2d"]
                elif "keypoints" in data:
                    keypoints = data["keypoints"]
            
            # Handle flat list format [x1, y1, conf1, x2, y2, conf2, ...]
            for i in range(0, len(keypoints), 3):
                if i + 2 >= len(keypoints):
                    break
                try:
                    x, y, conf = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                    if conf is None or conf < 0.1:
                        continue
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < W and 0 <= yi < H:
                        cv2.circle(pose_map, (xi, yi), 3, 255, -1)
                except (ValueError, TypeError, IndexError):
                    continue
                    
        except Exception as e:
            print(f"[warn] pose parse failed for {json_path.name}: {e}")
        
        return Image.fromarray(pose_map, mode="L")

    def _cache_item(self, idx):
        """Cache a single item in memory"""
        try:
            person_name, cloth_name = self.pairs[idx]
            person_path = self.person_dir / person_name
            cloth_path = self.cloth_dir / cloth_name
            mask_path = self._find_mask(self.mask_dir, cloth_name)
            pose_json = self.pose_dir / (Path(person_name).stem + "_keypoints.json")

            person_img = Image.open(person_path).convert("RGB")
            cloth_img = Image.open(cloth_path).convert("RGB")
            cloth_mask = Image.open(mask_path).convert("L") if mask_path and mask_path.exists() else Image.new("L", cloth_img.size, 255)
            pose_map = self._load_pose_map(pose_json, (person_img.size[1], person_img.size[0])) if pose_json.exists() else Image.new("L", person_img.size, 0)
            body_mask = Image.new("L", person_img.size, 255)
            
            self.cached_data[idx] = (person_img, cloth_img, cloth_mask, pose_map, body_mask)
        except Exception as e:
            print(f"Warning: cache item {idx} failed: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache_images and idx in self.cached_data:
            person_img, cloth_img, cloth_mask, pose_map, body_mask = self.cached_data[idx]
        else:
            person_name, cloth_name = self.pairs[idx]
            person_path = self.person_dir / person_name
            cloth_path = self.cloth_dir / cloth_name
            mask_path = self._find_mask(self.mask_dir, cloth_name)
            pose_json = self.pose_dir / (Path(person_name).stem + "_keypoints.json")

            if not person_path.exists():
                raise FileNotFoundError(f"Person not found: {person_path}")
            if not cloth_path.exists():
                raise FileNotFoundError(f"Cloth not found: {cloth_path}")

            person_img = Image.open(person_path).convert("RGB")
            cloth_img = Image.open(cloth_path).convert("RGB")
            cloth_mask = Image.open(mask_path).convert("L") if mask_path and mask_path.exists() else Image.new("L", cloth_img.size, 255)
            pose_map = self._load_pose_map(pose_json, (person_img.size[1], person_img.size[0])) if pose_json.exists() else Image.new("L", person_img.size, 0)
            body_mask = Image.new("L", person_img.size, 255)

        if self.transform is not None:
            person_img = self.transform(person_img)
            cloth_img = self.transform(cloth_img)
        if self.pose_transform is not None:
            cloth_mask = self.pose_transform(cloth_mask)
            pose_map = self.pose_transform(pose_map)
            body_mask = self.pose_transform(body_mask)

        inp = torch.cat([person_img, cloth_img, cloth_mask, pose_map, body_mask], dim=0)
        target = person_img
        return inp, target


# -------------------------
#   Warmup + Cosine scheduler
# -------------------------
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_lr = self.base_lrs.copy()

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = epoch / max(1, self.warmup_epochs)
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self.base_lrs[i] * scale
                self.last_lr[i] = g["lr"]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, (self.max_epochs - self.warmup_epochs))
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self.eta_min + (self.base_lrs[i] - self.eta_min) * cosine
                self.last_lr[i] = g["lr"]

    def get_last_lr(self):
        return self.last_lr


# -------------------------
#   Training
# -------------------------
def train_model(data_dir, output_dir, epochs=50, batch_size=4, lr=2e-4, resume_checkpoint=None,
               num_workers=4, cache_data=False, benchmark=False, save_every=10):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    pose_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = VirtualTryOnDataset(
        data_dir=data_dir, pairs_file="train_pairs.txt",
        transform=transform, pose_transform=pose_transform, split="train",
        cache_images=cache_data
    )
    test_dataset = VirtualTryOnDataset(
        data_dir=data_dir, pairs_file="test_pairs.txt",
        transform=transform, pose_transform=pose_transform, split="test",
        cache_images=cache_data
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset : {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model = TryOnGenerator(in_channels=9, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = WarmupCosineScheduler(optimizer, warmup_epochs=5, max_epochs=epochs, eta_min=1e-6)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    scaler = GradScaler()

    # Resume (fixed for PyTorch 2.6+)
    start_epoch = 0
    best = float("inf")
    best_weights = output_dir / "best_model.pth"

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        try:
            ckpt = load_checkpoint_trusted(resume_checkpoint, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "epoch" in ckpt:
                    start_epoch = int(ckpt["epoch"]) + 1
                if "best" in ckpt:
                    best = float(ckpt["best"])
                if "scaler" in ckpt:
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception:
                        pass
            else:
                model.load_state_dict(ckpt)
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("Continuing without resume...")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        sched.step(epoch)

        # --- Train ---
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                y_hat = model(x)
                loss = l1(y_hat, y) + 0.1 * mse(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validate ---
        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [test ]")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = l1(y_hat, y) + 0.1 * mse(y_hat, y)
                te_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss /= max(1, len(train_loader))
        te_loss /= max(1, len(test_loader))
        print(f"Epoch {epoch+1}: train={tr_loss:.4f}  test={te_loss:.4f}  lr={sched.get_last_lr()[0]:.6f}")

        # periodic checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best": best,
                "scaler": scaler.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")

        # best weights
        if te_loss < best:
            best = te_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best": best,
                "scaler": scaler.state_dict(),
            }, best_weights)
            print(f"  ↳ saved new best to {best_weights} (test {best:.4f})")

    print("Training completed!")
    print(f"Best test loss: {best:.4f}")
    return best_weights


# -------------------------
#   VirtualTryOnModel Class
# -------------------------
class VirtualTryOnModel:
    """
    Wrapper class for the virtual try-on model that handles loading,
    preprocessing and inference
    """
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = TryOnGenerator(in_channels=9, out_channels=3).to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = load_checkpoint_trusted(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model.load_state_dict(checkpoint["model"])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path}: {e}")
                print("Continuing with randomly initialized weights...")
        elif model_path:
            print(f"Warning: Model path {model_path} does not exist. Using randomly initialized weights.")
        
        self.model.eval()
        
        # Define transforms for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    @property
    def generator(self):
        """Backward compatibility property for code that expects 'generator'"""
        return self.model
    
    def _load_pose_map(self, pose_path, size_hw):
        """Load pose keypoints from JSON file and create pose map"""
        H, W = size_hw
        pose_map = np.zeros((H, W), dtype=np.uint8)
        
        try:
            with open(pose_path, "r") as f:
                data = json.load(f)
            
            keypoints = []
            if isinstance(data, dict):
                if "people" in data and data["people"]:
                    keypoints = data["people"][0].get("pose_keypoints_2d", [])
                elif "pose_keypoints_2d" in data:
                    keypoints = data["pose_keypoints_2d"]
                elif "keypoints" in data:
                    keypoints = data["keypoints"]
            
            # Handle different keypoint formats
            if isinstance(keypoints, list):
                # If keypoints is already a list of [x, y, conf] triplets
                if keypoints and isinstance(keypoints[0], list) and len(keypoints[0]) == 3:
                    for x, y, conf in keypoints:
                        if x is None or y is None or conf is None:
                            continue
                        try:
                            xi, yi = int(float(x)), int(float(y))
                            if 0 <= xi < W and 0 <= yi < H:
                                cv2.circle(pose_map, (xi, yi), 3, 255, -1)
                        except (ValueError, TypeError):
                            continue
                # If keypoints is a flat list [x1, y1, conf1, x2, y2, conf2, ...]
                else:
                    for i in range(0, len(keypoints), 3):
                        if i + 2 >= len(keypoints):
                            break
                        try:
                            x, y, conf = float(keypoints[i]), float(keypoints[i + 1]), float(keypoints[i + 2])
                            if conf < 0.1:
                                continue
                            xi, yi = int(round(x)), int(round(y))
                            if 0 <= xi < W and 0 <= yi < H:
                                cv2.circle(pose_map, (xi, yi), 3, 255, -1)
                        except (ValueError, TypeError, IndexError):
                            continue
                    
        except Exception as e:
            print(f"Warning: Failed to parse pose data from {pose_path}: {e}")
            
        return Image.fromarray(pose_map, mode="L")
    
    def _create_body_mask(self, person_img):
        """Create a basic body mask for the person image."""
        try:
            # Convert to numpy array
            img_array = np.array(person_img)
            h, w = img_array.shape[:2]
            
            # Use GrabCut for basic body segmentation
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Define rectangle around the center (assume person is centered)
            margin_x, margin_y = int(w * 0.1), int(h * 0.05)
            rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img_array, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return Image.fromarray(mask, mode='L')
            
        except Exception as e:
            print(f"Body mask creation failed: {e}")
            # Return a simple elliptical mask as fallback
            mask = Image.new('L', person_img.size, 0)
            draw = ImageDraw.Draw(mask)
            w, h = person_img.size
            center_x, center_y = w // 2, h // 2
            axes_x, axes_y = int(w * 0.3), int(h * 0.4)
            draw.ellipse([center_x - axes_x, center_y - axes_y, 
                         center_x + axes_x, center_y + axes_y], fill=255)
            return mask
    
    def try_on(self, person_image_path, cloth_image_path, pose_json_path=None, cloth_mask_path=None):
        """
        Perform try-on using the trained model with proper pose integration.
        
        Args:
            person_image_path: Path to the person image
            cloth_image_path: Path to the cloth image
            pose_json_path: Path to the pose JSON file (optional)
            cloth_mask_path: Path to the cloth mask (optional)
        
        Returns:
            PIL Image of the try-on result
        """
        # Load images
        person_img = Image.open(person_image_path).convert("RGB")
        cloth_img = Image.open(cloth_image_path).convert("RGB")
        cloth_mask = Image.open(cloth_mask_path).convert("L") if cloth_mask_path and os.path.exists(cloth_mask_path) else None

        # Load pose JSON and create proper pose map
        pose_map = None
        if pose_json_path and os.path.exists(pose_json_path):
            try:
                with open(pose_json_path, 'r') as f:
                    pose_data = json.load(f)
                keypoints = pose_data.get("keypoints", [])
                
                if keypoints:
                    # Create pose map image
                    pose_map_img = Image.new("L", person_img.size, 0)
                    draw = ImageDraw.Draw(pose_map_img)
                    
                    # Draw connections between keypoints for better pose representation
                    pose_connections = [
                        (5, 6),   # shoulders
                        (5, 7),   # left shoulder to elbow
                        (6, 8),   # right shoulder to elbow
                        (7, 9),   # left elbow to wrist
                        (8, 10),  # right elbow to wrist
                        (11, 12), # hips
                        (5, 11),  # left shoulder to hip
                        (6, 12),  # right shoulder to hip
                        (11, 13), # left hip to knee
                        (12, 14), # right hip to knee
                        (13, 15), # left knee to ankle
                        (14, 16)  # right knee to ankle
                    ]
                    
                    # Draw keypoints
                    for i, kp in enumerate(keypoints):
                        if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                            x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                            if conf >= 0.1 and 0 <= x < person_img.size[0] and 0 <= y < person_img.size[1]:
                                # Draw circle for keypoint
                                draw.ellipse((x-4, y-4, x+4, y+4), fill=255)
                    
                    # Draw connections
                    for conn in pose_connections:
                        if len(keypoints) > max(conn):
                            kp1, kp2 = keypoints[conn[0]], keypoints[conn[1]]
                            if (len(kp1) >= 3 and len(kp2) >= 3 and 
                                kp1[2] >= 0.1 and kp2[2] >= 0.1):
                                x1, y1 = int(kp1[0]), int(kp1[1])
                                x2, y2 = int(kp2[0]), int(kp2[1])
                                draw.line((x1, y1, x2, y2), fill=128, width=2)
                    
                    pose_map = pose_map_img
                    
            except Exception as e:
                print(f"Warning: Could not load pose data from {pose_json_path}: {e}")
                pose_map = None

        # Run the model prediction with enhanced processing
        try:
            result_img = self.predict(person_img, cloth_img, cloth_mask, pose_map)
            
            # Post-process the result for better quality
            result_img = self._enhance_result(result_img, person_img, cloth_img, cloth_mask)
            
            return result_img
            
        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Fallback to enhanced warping method
            return self._fallback_try_on(person_img, cloth_img, cloth_mask, pose_map)
    
    def _enhance_result(self, result_img, person_img, cloth_img, cloth_mask):
        """Enhance the model output for better visual quality."""
        try:
            # Convert to numpy for processing
            result_array = np.array(result_img)
            person_array = np.array(person_img.resize(result_img.size))
            
            # Apply sharpening filter to improve details
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(result_array, -1, kernel)
            
            # Blend original person in areas where cloth is not present
            if cloth_mask:
                mask_array = np.array(cloth_mask.resize(result_img.size)) / 255.0
                mask_3d = np.stack([mask_array] * 3, axis=-1)
                
                # Keep person's face and other non-cloth areas
                face_region = mask_3d < 0.3  # Areas where cloth shouldn't be
                result_array = np.where(face_region, person_array, sharpened)
            else:
                result_array = sharpened
            
            # Ensure values are in valid range
            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
            
            return Image.fromarray(result_array)
            
        except Exception as e:
            print(f"Result enhancement failed: {e}")
            return result_img
    
    def _fallback_try_on(self, person_img, cloth_img, cloth_mask, pose_map):
        """Fallback try-on method using intelligent warping and blending."""
        try:
            # Resize all images to consistent size
            target_size = (512, 512)
            person_resized = person_img.resize(target_size)
            cloth_resized = cloth_img.resize(target_size)
            
            # Create warped cloth based on body proportions
            warped_cloth = self._intelligent_warp(cloth_resized, person_resized, pose_map)
            
            # Create blending mask
            if cloth_mask:
                blend_mask = cloth_mask.resize(target_size)
            else:
                # Create simple torso mask
                blend_mask = Image.new('L', target_size, 0)
                draw = ImageDraw.Draw(blend_mask)
                w, h = target_size
                # Draw torso region
                draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)
                draw.ellipse([w//3, h//4, 2*w//3, h//2], fill=255)
            
            # Apply Gaussian blur to mask for smoother blending
            mask_array = np.array(blend_mask)
            mask_blurred = cv2.GaussianBlur(mask_array, (15, 15), 0)
            mask_normalized = mask_blurred / 255.0
            
            # Blend images
            person_array = np.array(person_resized)
            warped_array = np.array(warped_cloth)
            
            mask_3d = np.stack([mask_normalized] * 3, axis=-1)
            result_array = (warped_array * mask_3d + person_array * (1 - mask_3d)).astype(np.uint8)
            
            return Image.fromarray(result_array)
            
        except Exception as e:
            print(f"Fallback try-on failed: {e}")
            # Final fallback - simple blend
            return Image.blend(person_img.resize((512, 512)), cloth_img.resize((512, 512)), 0.4)
    
    def _intelligent_warp(self, cloth_img, person_img, pose_map):
        """Intelligently warp cloth to fit person's body shape."""
        try:
            cloth_array = np.array(cloth_img)
            h, w = cloth_array.shape[:2]
            
            # If pose map is available, use it for warping
            if pose_map:
                pose_array = np.array(pose_map.resize((w, h)))
                
                # Find key body points from pose map
                y_indices, x_indices = np.where(pose_array > 128)
                
                if len(x_indices) > 0 and len(y_indices) > 0:
                    # Calculate body bounds
                    body_left = max(0, np.min(x_indices) - 20)
                    body_right = min(w, np.max(x_indices) + 20)
                    body_top = max(0, np.min(y_indices) - 10)
                    body_bottom = min(h, np.max(y_indices) + 10)
                    
                    # Create perspective transformation
                    cloth_corners = np.array([
                        [0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]
                    ], dtype=np.float32)
                    
                    body_corners = np.array([
                        [body_left, body_top],
                        [body_right, body_top],
                        [body_right, body_bottom],
                        [body_left, body_bottom]
                    ], dtype=np.float32)
                    
                    # Apply perspective transform
                    M = cv2.getPerspectiveTransform(cloth_corners, body_corners)
                    warped = cv2.warpPerspective(cloth_array, M, (w, h))
                    
                    return Image.fromarray(warped)
            
            # Default: simple scaling to fit torso area
            scale_factor = 0.8
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            # Resize and center
            cloth_scaled = cv2.resize(cloth_array, (new_w, new_h))
            
            # Create centered image
            result = np.zeros_like(cloth_array)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = cloth_scaled
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"Intelligent warping failed: {e}")
            return cloth_img

    def predict(self, person_img, cloth_img, cloth_mask=None, pose_map=None, body_mask=None):
        """Enhanced prediction with proper input handling."""
        # Ensure images are the correct size
        person_img = person_img.resize((512, 512))
        cloth_img = cloth_img.resize((512, 512))
        
        # Handle cloth mask
        if cloth_mask is None:
            cloth_mask = Image.new('L', (512, 512), 255)
        else:
            cloth_mask = cloth_mask.resize((512, 512))
        
        # Handle pose map - create a proper pose map if none provided
        if pose_map is None:
            pose_map = Image.new('L', (512, 512), 0)
        else:
            pose_map = pose_map.resize((512, 512))
        
        # Handle body mask - create a reasonable body mask if none provided
        if body_mask is None:
            # Create a better body mask using segmentation
            body_mask = self._create_body_mask(person_img)
        else:
            body_mask = body_mask.resize((512, 512))
        
        # Convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        person_tensor = transform(person_img).unsqueeze(0)
        cloth_tensor = transform(cloth_img).unsqueeze(0)
        cloth_mask_tensor = mask_transform(cloth_mask).unsqueeze(0)
        pose_tensor = mask_transform(pose_map).unsqueeze(0)
        body_mask_tensor = mask_transform(body_mask).unsqueeze(0)
        
        # Combine inputs for the model (person + cloth + cloth_mask + pose + body_mask)
        model_input = torch.cat([
            person_tensor,      # 3 channels
            cloth_tensor,       # 3 channels  
            cloth_mask_tensor,  # 1 channel
            pose_tensor,        # 1 channel
            body_mask_tensor    # 1 channel
        ], dim=1)  # Total: 9 channels
        
        # Ensure we're in evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            model_input = model_input.to(self.device)
            
            # Generate the output
            output = self.model(model_input)
            
            # Move back to CPU and convert to image
            output = output.cpu()
            
            # Denormalize
            output = (output + 1) / 2.0
            output = torch.clamp(output, 0, 1)
            
            # Convert to PIL Image
            output_np = output[0].permute(1, 2, 0).numpy()
            output_img = Image.fromarray((output_np * 255).astype(np.uint8))
            
            return output_img


# -------------------------
#   Backward Compatibility Alias
# -------------------------
TryOnModel = VirtualTryOnModel
