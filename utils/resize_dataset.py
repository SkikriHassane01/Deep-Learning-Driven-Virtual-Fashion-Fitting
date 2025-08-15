# created 14/08/25

from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# -------- SETTINGS --------
DATA_ROOT = Path(r"D:\03. Projects\technocolabs_projects\VirtualFashion\data")
IMAGE_SIZE = (512, 512)
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SKIP_EXTS = {".json", ".txt"}
# --------------------------


def resize_image_in_place(path: Path):
    """Resize image in-place, overwriting the original file."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
        im.save(path, quality=95)


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"❌ DATA_ROOT not found: {DATA_ROOT.resolve()}")

    files = [p for p in DATA_ROOT.rglob("*") if p.is_file()]
    tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in files:
            ext = f.suffix.lower()
            if ext in IMG_EXTS:
                tasks.append(ex.submit(resize_image_in_place, f))
            elif ext in SKIP_EXTS:
                continue
            else:
                continue

        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Resizing in-place"):
            pass

    print(f"\n✅ Done. All images resized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} in-place.")
    print("JSON and TXT files were left untouched.")


if __name__ == "__main__":
    main()
