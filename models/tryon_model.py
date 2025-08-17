import os
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


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
    def __init__(self, data_dir, pairs_file, transform=None, pose_transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split  # "train" or "test"
        self.transform = transform
        self.pose_transform = pose_transform

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


# -------------------------
#   Training
# -------------------------
def train_model(data_dir, output_dir, epochs=50, batch_size=4, lr=2e-4):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 512x512 everywhere (simple)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    pose_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_ds = VirtualTryOnDataset(
        data_dir=data_dir,
        pairs_file="train_pairs.txt",
        transform=transform,
        pose_transform=pose_transform,
        split="train"
    )
    test_ds = VirtualTryOnDataset(
        data_dir=data_dir,
        pairs_file="test_pairs.txt",
        transform=transform,
        pose_transform=pose_transform,
        split="test"
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Test  size: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    model = TryOnGenerator(in_channels=9, out_channels=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    best = float("inf")
    best_weights = output_dir / "best_model_weights.pth"

    for epoch in range(epochs):
        # ---- train
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            y_hat = model(x)
            loss_l1 = l1(y_hat, y)
            loss_mse = mse(y_hat, y)
            loss = loss_l1 + 0.1 * loss_mse
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- val
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
        sched.step()

        print(f"Epoch {epoch+1}: train={tr_loss:.4f}  test={te_loss:.4f}  lr={sched.get_last_lr()[0]:.6f}")

        # save best weights
        if te_loss < best:
            best = te_loss
            torch.save(model.state_dict(), best_weights)
            print(f"  ↳ saved new best to {best_weights} (test {best:.4f})")

        # periodic
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    print("Done.")
    print(f"Best test loss: {best:.4f}")
    return best_weights
