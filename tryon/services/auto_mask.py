from pathlib import Path
from PIL import Image, ImageDraw

def auto_cloth_mask(cloth_img_path, mask_output_path):
    """
    Automatically generate cloth mask from cloth image.
    
    Args:
        cloth_img_path: Path to the cloth image
        mask_output_path: Path where the mask should be saved
        
    Returns:
        Path: Path to the generated mask
    """
    try:
        with Image.open(cloth_img_path) as cloth_img:
            # Convert to grayscale for simple thresholding
            gray = cloth_img.convert('L')
            
            # Create mask based on image content
            # Simple approach: assume non-white areas are cloth
            mask = Image.new('L', cloth_img.size, 0)
            
            # Get image data
            pixels = list(gray.getdata())
            mask_pixels = []
            
            # Simple thresholding: if pixel is not too bright, consider it cloth
            threshold = 240
            for pixel in pixels:
                if pixel < threshold:
                    mask_pixels.append(255)  # cloth area (white)
                else:
                    mask_pixels.append(0)    # background (black)
            
            mask.putdata(mask_pixels)
            
            # Apply some morphological operations to clean up the mask
            # For now, just save as is
            mask.save(mask_output_path)
            
    except Exception as e:
        print(f"Error creating auto mask: {e}")
        # Fallback: create a simple rectangular mask
        try:
            with Image.open(cloth_img_path) as cloth_img:
                mask = Image.new('L', cloth_img.size, 255)  # All white
                mask.save(mask_output_path)
        except:
            # Final fallback
            mask = Image.new('L', (512, 512), 255)
            mask.save(mask_output_path)
    
    return mask_output_path
