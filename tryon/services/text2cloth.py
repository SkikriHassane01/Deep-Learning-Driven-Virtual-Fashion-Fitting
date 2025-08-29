# tryon/services/text2cloth.py
from pathlib import Path
from PIL import Image, ImageDraw
import random

SIZE = (768, 1024)
TORSO_RECT = (110, 200, 658, 820)

COLOR_MAP = {
    'red': (205, 40, 60), 'blue': (50, 90, 200), 'green': (40, 140, 80),
    'black': (25, 25, 25), 'white': (235, 235, 235), 'yellow': (230, 200, 50),
    'purple': (120, 70, 160), 'pink': (230, 110, 160), 'brown': (120, 80, 40),
    'orange': (240, 140, 60), 'gray': (140, 140, 140),
}

def _pick_color(text: str):
    t = text.lower()
    for k, v in COLOR_MAP.items():
        if k in t: return v
    return (random.randint(60, 200), random.randint(60, 200), random.randint(60, 200))

def _apply_pattern(draw: ImageDraw.ImageDraw, rect, text: str):
    x1, y1, x2, y2 = rect
    t = text.lower()
    if 'stripe' in t:
        for i in range(y1, y2, 24):
            draw.rectangle([x1, i, x2, i+8], fill=(255,255,255,40))
    elif 'dot' in t or 'polka' in t:
        for yy in range(y1+12, y2, 36):
            for xx in range(x1+12, x2, 36):
                draw.ellipse([xx-6, yy-6, xx+6, yy+6], fill=(255,255,255,60))
    elif 'check' in t or 'plaid' in t:
        for i in range(x1, x2, 40):
            draw.rectangle([i, y1, i+6, y2], fill=(255,255,255,50))
        for j in range(y1, y2, 40):
            draw.rectangle([x1, j, x2, j+6], fill=(255,255,255,50))
    elif 'floral' in t:
        for _ in range(120):
            cx = random.randint(x1, x2); cy = random.randint(y1, y2)
            r = random.randint(5, 12)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255,255,255,90), width=2)

def generate_cloth_from_prompt(prompt: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    base = Image.new("RGBA", SIZE, (0,0,0,0))
    draw = ImageDraw.Draw(base)

    color = _pick_color(prompt)
    x1, y1, x2, y2 = TORSO_RECT
    draw.rounded_rectangle(TORSO_RECT, radius=40, fill=(*color, 255))
    draw.ellipse([x1+180, y1-80, x2-180, y1+60], fill=(0,0,0,0))
    draw.polygon([(x1, y1+120), (x1-120, y1+220), (x1, y1+300)], fill=(*color, 255))
    draw.polygon([(x2, y1+120), (x2+120, y1+220), (x2, y1+300)], fill=(*color, 255))
    _apply_pattern(draw, TORSO_RECT, prompt)

    cloth_img_path = out / "cloth_from_text.png"
    base.save(cloth_img_path)

    # Save mask (alpha must exist)
    alpha = base.split()[-1]
    if alpha.getextrema() == (0, 0):
        raise RuntimeError("Generated cloth has empty alpha mask.")
    mask_path = out / "cloth_from_text_mask.png"
    alpha.save(mask_path)

    return str(cloth_img_path), str(mask_path)
