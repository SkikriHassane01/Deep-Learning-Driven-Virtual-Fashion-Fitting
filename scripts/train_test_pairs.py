from pathlib import Path

root = Path("data/DeepFashion")
for split in ("train", "test"):
    img_dir  = root/split/"image"
    cloth_dir= root/split/"cloth"
    out_txt  = root/f"{split}_pairs.txt"
    with out_txt.open("w") as f:
        for img_path in img_dir.rglob("*.jpg"):
            rel_img   = img_path.relative_to(img_dir).as_posix()
            cloth_png = (cloth_dir/rel_img).with_suffix(".png")
            if cloth_png.exists():
                f.write(f"{rel_img}  {rel_img[:-4]}.png\n")
