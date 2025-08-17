#!/usr/bin/env python3
"""
Minimal setup + training runner for Virtual Fashion Try-On.

- Validates *your* folder structure:
    data/{train,test}/{image,cloth,cloth-mask,openpose_json}/
    data/train_pairs.txt, data/test_pairs.txt

- Starts training using try_on_model.train_model
"""
from pathlib import Path
import argparse
import sys

from models.tryon_model import train_model


REQUIRED_SPLIT_DIRS = ["image", "cloth", "cloth-mask", "openpose_json"]


def read_pairs(pairs_path: Path):
    pairs = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line in {pairs_path}: {line}")
            pairs.append((parts[0], parts[1]))
    return pairs


def validate_data(data_dir: Path) -> bool:
    ok = True
    # files
    train_pairs = data_dir / "train_pairs.txt"
    test_pairs  = data_dir / "test_pairs.txt"
    if not train_pairs.exists():
        print(f"❌ Missing file: {train_pairs}")
        ok = False
    if not test_pairs.exists():
        print(f"❌ Missing file: {test_pairs}")
        ok = False
    if not ok:
        return False

    # dirs
    for split in ["train", "test"]:
        for d in REQUIRED_SPLIT_DIRS:
            p = data_dir / split / d
            if not p.exists():
                print(f"❌ Missing directory: {p}")
                ok = False
    if not ok:
        return False

    # sample existence check for a few pairs
    def check_pairs(split: str, pairs_file: Path):
        pairs = read_pairs(pairs_file)
        if not pairs:
            print(f"❌ No pairs in {pairs_file}")
            return False
        # check first 5
        for i, (person, cloth) in enumerate(pairs[:5]):
            person_path = data_dir / split / "image" / person
            cloth_path  = data_dir / split / "cloth" / cloth
            if not person_path.exists():
                print(f"❌ Person image not found: {person_path}")
                return False
            if not cloth_path.exists():
                print(f"❌ Cloth image not found: {cloth_path}")
                return False
        print(f"✅ {pairs_file.name}: {len(pairs)} pairs found")
        return True

    ok &= check_pairs("train", train_pairs)
    ok &= check_pairs("test",  test_pairs)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser("Setup & Train (simple)")
    ap.add_argument("--data-dir", default="data", type=str)
    ap.add_argument("--output-dir", default="checkpoints", type=str)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)

    print("📊 Validating data structure...")
    if not validate_data(data_dir):
        print("❌ Data validation failed")
        return 1
    print("✅ Data looks good\n")

    if args.validate_only:
        print("✅ Validation-only mode. Nothing else to do.")
        return 0

    print("🏋️ Starting training...")
    best_path = train_model(
        data_dir=data_dir,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    print(f"\n🎉 Training finished. Best weights at: {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
