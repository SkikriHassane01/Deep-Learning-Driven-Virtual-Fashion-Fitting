# tryon/services/auto_mask.py
from pathlib import Path
from PIL import Image, ImageOps

def auto_cloth_mask(cloth_image_path, mask_output_path):
    """
    Enhanced cloth mask generation:
    - Use alpha channel if present; otherwise apply advanced background removal.
    Always writes a mask file. Accepts Path or str.
    """
    cloth_image_path = Path(cloth_image_path)
    mask_output_path = Path(mask_output_path)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(cloth_image_path).convert("RGBA") as im:
        alpha = im.getchannel("A")
        
        # If alpha channel is helpful, use it
        if alpha.getextrema() != (255, 255):
            mask = alpha
        else:
            # Enhanced background removal using multiple methods
            rgb_img = im.convert("RGB")
            
            # Method 1: HSV-based background removal
            import numpy as np
            import cv2
            
            img_array = np.array(rgb_img)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Create mask for non-white backgrounds
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Invert to get foreground
            fg_mask = cv2.bitwise_not(white_mask)
            
            # Clean up with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Fill holes
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)
            
            # If mask is too empty or too full, use GrabCut
            mask_ratio = np.sum(fg_mask > 0) / fg_mask.size
            if mask_ratio < 0.1 or mask_ratio > 0.9:
                print("HSV mask poor quality, trying GrabCut...")
                
                # Use GrabCut for better segmentation
                mask_gc = np.zeros(img_array.shape[:2], dtype=np.uint8)
                h, w = img_array.shape[:2]
                
                # Define rectangle (assume cloth is in center 80% of image)
                margin_x, margin_y = int(w * 0.1), int(h * 0.1)
                rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
                
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                
                try:
                    cv2.grabCut(img_array, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                    fg_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')
                except:
                    print("GrabCut failed, using HSV mask")
            
            mask = Image.fromarray(fg_mask, mode="L")

        # Ensure mask is binary and clean
        mask = mask.convert("L")
        # Apply threshold to ensure binary mask
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_array, mode="L")
        
        mask.save(mask_output_path)

    return str(mask_output_path)
