from pathlib import Path
import json
from PIL import Image
import numpy as np

def process_user_image(user_image_path, out_dir):
    """
    Process user image to extract pose and body mask.
    
    Args:
        user_image_path: Path to the user's uploaded image
        out_dir: Output directory for generated files
        
    Returns:
        tuple: (mask_path, pose_json_path)
    """
    mask_path = out_dir / "body_mask.png"
    pose_json_path = out_dir / "pose.json"
    
    # Create a simple body mask (full body silhouette)
    try:
        with Image.open(user_image_path) as img:
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create a simple mask (white foreground, black background)
            mask = Image.new('L', img.size, 255)  # White mask for now
            mask.save(mask_path)
            
            # Create basic pose data
            w, h = img.size
            pose_data = {
                "keypoints": [
                    [w//2, h//4],    # head
                    [w//2, h//2],    # torso center
                    [w//4, h//3],    # left shoulder
                    [3*w//4, h//3],  # right shoulder
                    [w//4, 2*h//3],  # left hip
                    [3*w//4, 2*h//3] # right hip
                ],
                "image_size": [w, h]
            }
            
            with open(pose_json_path, 'w') as f:
                json.dump(pose_data, f)
                
    except Exception as e:
        print(f"Error processing image: {e}")
        # Fallback: create minimal files
        mask_path.touch()
        with open(pose_json_path, 'w') as f:
            json.dump({"keypoints": [], "image_size": [512, 512]}, f)
    
    return mask_path, pose_json_path
