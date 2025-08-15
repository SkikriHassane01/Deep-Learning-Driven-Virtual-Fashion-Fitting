from pathlib import Path
import shutil
from PIL import Image, ImageDraw

def baseline_fit(user_img_path, pose_json_path, cloth_img_path, cloth_mask_path, out_path):
    """
    Perform baseline virtual try-on fitting.
    
    Args:
        user_img_path: Path to user image
        pose_json_path: Path to pose JSON file
        cloth_img_path: Path to cloth image
        cloth_mask_path: Path to cloth mask (optional)
        out_path: Output path for result image
    """
    try:
        # Open user image
        with Image.open(user_img_path) as user_img:
            result = user_img.copy()
            
            # If cloth image exists, try to overlay it
            if cloth_img_path and cloth_img_path.exists():
                with Image.open(cloth_img_path) as cloth_img:
                    # Resize cloth to fit user torso area
                    w, h = user_img.size
                    cloth_size = (w//2, h//3)
                    cloth_resized = cloth_img.resize(cloth_size, Image.Resampling.LANCZOS)
                    
                    # Position cloth on torso area
                    paste_x = w//4
                    paste_y = h//3
                    
                    # Simple paste (overlay)
                    if cloth_resized.mode == 'RGBA':
                        result.paste(cloth_resized, (paste_x, paste_y), cloth_resized)
                    else:
                        result.paste(cloth_resized, (paste_x, paste_y))
            
            # Save result
            result.save(out_path, 'JPEG', quality=95)
            
    except Exception as e:
        print(f"Error in baseline_fit: {e}")
        # Fallback: copy user image
        shutil.copy2(user_img_path, out_path)
