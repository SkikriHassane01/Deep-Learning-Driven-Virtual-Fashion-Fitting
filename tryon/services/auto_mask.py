# tryon/services/auto_mask.py
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

class ClothMaskError(RuntimeError):
    pass

def auto_cloth_mask(cloth_image_path, target_mask_path):
    """
    Extracts mask strictly from the image alpha channel.
    If alpha is missing/empty -> raises ClothMaskError.
    """
    cloth_image_path = Path(cloth_image_path)
    target_mask_path = Path(target_mask_path)

    im = Image.open(cloth_image_path).convert("RGBA")
    alpha = np.array(im.split()[-1])

    if alpha.max() == 0:
        raise ClothMaskError("Cloth image contains no alpha channel content to form a mask.")

    mask = (alpha > 0).astype(np.uint8) * 255
    target_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_mask_path), mask)
    return str(target_mask_path)
