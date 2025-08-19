from pathlib import Path
import json
import cv2
import numpy as np
import torch
from PIL import Image
import os
import shutil
import sys

# Add parent directories to path to find models module
current_dir = Path(__file__).parent  # services
parent_dir = current_dir.parent.parent.parent  # project root
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import our virtual try-on model
try:
    from models.tryon_model import VirtualTryOnModel
except ImportError:
    print("Warning: Could not import VirtualTryOnModel. Check project structure.")

# Lazy-load the try-on model
_tryon_model = None

def _get_tryon_model():
    global _tryon_model
    if _tryon_model is None:
        # Look for trained model in the models directory
        model_dir = Path(__file__).parent.parent.parent / "models"
        model_path = model_dir / "tryon" / "best_model.pth"
        
        if model_path.exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _tryon_model = VirtualTryOnModel(model_path=model_path, device=device)
            print(f"Loaded try-on model from {model_path}")
        else:
            # Use default model (will be initialized without pre-trained weights)
            _tryon_model = VirtualTryOnModel()
            print("Using untrained try-on model (no pre-trained weights found)")
    
    return _tryon_model

def warping_cloth(cloth, cloth_mask, pose_data, target_size):
    """
    Warp clothing based on pose data to fit the body.
    
    Args:
        cloth: Path to cloth image
        cloth_mask: Path to cloth mask
        pose_data: JSON data with pose keypoints
        target_size: Target size (width, height) for warped cloth
        
    Returns:
        Warped cloth image
    """
    try:
        # Load cloth and mask
        cloth_img = cv2.imread(str(cloth))
        mask_img = cv2.imread(str(cloth_mask), cv2.IMREAD_GRAYSCALE)
        
        if cloth_img is None or mask_img is None:
            raise ValueError("Could not load cloth or mask image")
        
        # Parse pose data for key points
        with open(pose_data, 'r') as f:
            pose = json.load(f)
        
        keypoints = pose.get("keypoints", [])
        
        # Get target points for warping (shoulders and hips)
        # These indices might need adjustment based on your pose model
        shoulder_left = keypoints[2] if len(keypoints) > 2 else [target_size[0]//3, target_size[1]//3, 1.0]
        shoulder_right = keypoints[3] if len(keypoints) > 3 else [2*target_size[0]//3, target_size[1]//3, 1.0]
        hip_left = keypoints[9] if len(keypoints) > 9 else [target_size[0]//3, 2*target_size[1]//3, 1.0]
        hip_right = keypoints[10] if len(keypoints) > 10 else [2*target_size[0]//3, 2*target_size[1]//3, 1.0]
        
        # Source points (corners of the cloth)
        h, w = cloth_img.shape[:2]
        src_points = np.array([
            [0, 0],               # Top-left
            [w, 0],               # Top-right
            [0, h],               # Bottom-left
            [w, h]                # Bottom-right
        ], dtype=np.float32)
        
        # Target points (map to body shape)
        dst_points = np.array([
            [shoulder_left[0], shoulder_left[1]],       # Top-left
            [shoulder_right[0], shoulder_right[1]],     # Top-right
            [hip_left[0], hip_left[1]],                 # Bottom-left
            [hip_right[0], hip_right[1]]                # Bottom-right
        ], dtype=np.float32)
        
        # Get perspective transform and apply it
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_cloth = cv2.warpPerspective(cloth_img, M, target_size)
        warped_mask = cv2.warpPerspective(mask_img, M, target_size)
        
        return warped_cloth, warped_mask
        
    except Exception as e:
        print(f"Error in warping_cloth: {e}")
        # Return originals if warping fails
        return cv2.resize(cloth_img, target_size), cv2.resize(mask_img, target_size)

def baseline_fit(user_img_path, pose_json_path, cloth_img_path, cloth_mask_path, out_path):
    """
    Perform virtual try-on fitting using our model or a baseline method.
    
    Args:
        user_img_path: Path to user image
        pose_json_path: Path to pose JSON file
        cloth_img_path: Path to cloth image
        cloth_mask_path: Path to cloth mask
        out_path: Output path for result image
    """
    # First attempt to use our trained try-on model
    try:
        model = _get_tryon_model()
        result_img = model.try_on(user_img_path, cloth_img_path, pose_json_path, cloth_mask_path)
        result_img.save(out_path)
        print("Try-on completed with trained model")
        return
    except Exception as model_error:
        print(f"Try-on model failed: {model_error}. Falling back to baseline method.")
    
    # Fallback to basic warping and blending if the model fails
    try:
        # Load user image
        user_img = cv2.imread(str(user_img_path))
        if user_img is None:
            raise ValueError(f"Could not read image at {user_img_path}")
        
        # Get target size
        target_size = (user_img.shape[1], user_img.shape[0])
        
        # Warp cloth to fit pose
        warped_cloth, warped_mask = warping_cloth(
            cloth_img_path, 
            cloth_mask_path, 
            pose_json_path, 
            target_size
        )
        
        # Normalize mask to binary (0 or 255)
        _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create inverse mask for blending
        inv_mask = cv2.bitwise_not(warped_mask)
        
        # Split into RGB channels
        b, g, r = cv2.split(user_img)
        cb, cg, cr = cv2.split(warped_cloth)
        
        # Create mask with 3 channels for blending
        mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask])
        inv_mask_3ch = cv2.merge([inv_mask, inv_mask, inv_mask])
        
        # Scale masks to 0-1 range
        mask_3ch = mask_3ch.astype(float) / 255
        inv_mask_3ch = inv_mask_3ch.astype(float) / 255
        
        # Blend images
        result = user_img.astype(float) * inv_mask_3ch + warped_cloth.astype(float) * mask_3ch
        
        # Convert back to 8-bit image
        result = result.astype(np.uint8)
        
        # Save result
        cv2.imwrite(str(out_path), result)
        print("Try-on completed with baseline warping")
        
    except Exception as e:
        print(f"Error in baseline_fit: {e}")
        # Final fallback: simple overlay
        try:
            user_img = Image.open(user_img_path).convert("RGBA")
            cloth_img = Image.open(cloth_img_path).convert("RGBA")
            
            # Resize cloth to fit user image
            cloth_img = cloth_img.resize(
                (user_img.width // 2, user_img.height // 3),
                Image.LANCZOS
            )
            
            # Position cloth on torso area
            user_img_copy = user_img.copy()
            user_img_copy.paste(
                cloth_img,
                (user_img.width // 4, user_img.height // 3),
                cloth_img if cloth_img.mode == 'RGBA' else None
            )
            
            # Save result
            user_img_copy.convert("RGB").save(out_path, 'JPEG', quality=95)
            print("Try-on completed with simple overlay")
            
        except Exception as final_error:
            print(f"Final fallback failed: {final_error}")
            # Absolute last resort: just save the user image
            shutil.copy2(user_img_path, out_path)
