from pathlib import Path
from PIL import Image
import numpy as np
import cv2

def auto_cloth_mask(cloth_img_path, mask_output_path):
    """
    Automatically generate cloth mask from cloth image using GrabCut algorithm.
    
    Args:
        cloth_img_path: Path to the cloth image
        mask_output_path: Path where the mask should be saved
        
    Returns:
        Path: Path to the generated mask
    """
    try:
        # Read the image
        img = cv2.imread(str(cloth_img_path))
        if img is None:
            raise ValueError(f"Could not read image at {cloth_img_path}")
        
        # Create a simple initial mask
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # Assume the center part of the image is foreground
        h, w = img.shape[:2]
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 1, -1)
        
        # Background and foreground models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        rect = (w//4, h//4, w//2, h//2)
        try:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            # Fallback if GrabCut fails
            # Simple thresholding on lightness
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, mask = cv2.threshold(hsv[:,:,2], 200, 255, cv2.THRESH_BINARY_INV)
        
        # Create mask where sure and probable foreground are set to white
        mask2 = np.where((mask==2) | (mask==0), 0, 255).astype('uint8')
        
        # Clean up mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        
        # Further clean small noise by finding largest contour
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            mask2 = np.zeros_like(mask2)
            cv2.drawContours(mask2, [max_contour], -1, 255, -1)
        
        # Save the mask
        cv2.imwrite(str(mask_output_path), mask2)
        
    except Exception as e:
        print(f"Error in auto_cloth_mask: {e}")
        # Fallback: create a simple mask
        try:
            with Image.open(cloth_img_path) as img:
                # Convert to grayscale
                gray = img.convert('L')
                
                # Threshold: non-white pixels are considered cloth
                threshold = 240
                binary = gray.point(lambda x: 0 if x > threshold else 255, '1')
                binary.save(mask_output_path)
        except Exception as e2:
            print(f"Fallback mask creation failed: {e2}")
            # Final fallback: create a white rectangle
            from PIL import ImageDraw
            mask = Image.new('L', (512, 512), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([100, 100, 412, 412], fill=255)
            mask.save(mask_output_path)
    
    return Path(mask_output_path)
