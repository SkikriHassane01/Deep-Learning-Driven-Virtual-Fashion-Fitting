#!/usr/bin/env python3
from pathlib import Path
import shutil, sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please run: pip install huggingface_hub")
    sys.exit(1)

BASE = Path("models")

# Hugging Face repos & filenames (COCO variant of OpenPose)
OPENPOSE_REPO = "camenduru/openpose"
COCO_PROTOTXT = "models/pose/coco/pose_deploy_linevec.prototxt"
COCO_CAFFE   = "models/pose/coco/pose_iter_440000.caffemodel"

# U2Net mirror
U2NET_REPO = "flashingtt/U-2-Net"
U2NET_FILE = "u2net.pth"

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def fetch(repo_id: str, filename: str, dst: Path):
    if dst.exists():
        print(f"✅ {dst} already exists")
        return
    print(f"⬇️  {filename} -> {dst}")
    cache_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ensure_parent(dst)
    shutil.copy2(cache_path, dst)
    print(f"✅ Saved: {dst}")

def main():
    print(f"Downloading model files into: {BASE.resolve()}\n")

    # We place both OpenPose files under models/pose/ to keep paths simple
    prototxt_dst = BASE / "pose" / "pose_deploy_linevec.prototxt"
    caffemodel_dst = BASE / "pose" / "pose_iter_440000.caffemodel"
    u2net_dst = BASE / "segmentation" / "u2net.pth"

    # OpenPose COCO prototxt & weights
    fetch(OPENPOSE_REPO, COCO_PROTOTXT, prototxt_dst)
    fetch(OPENPOSE_REPO, COCO_CAFFE,   caffemodel_dst)

    # U2Net weights (used if you add portrait/background masking)
    fetch(U2NET_REPO, U2NET_FILE, u2net_dst)

    print("\n🎉 Done. All required files are present.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)
