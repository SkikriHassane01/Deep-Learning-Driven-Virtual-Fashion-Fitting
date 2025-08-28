from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import os
import shutil
import sys
from django.conf import settings
import json
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

def _get_tryon_model():
    global _tryon_model
    if _tryon_model is None:
        # Look for trained model in the models/tryon directory
        model_path = Path(__file__).parent.parent.parent / "models" / "tryon" / "best_model.pth"
        
        if model_path.exists():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                from models.tryon_model import VirtualTryOnModel
                _tryon_model = VirtualTryOnModel(model_path=model_path, device=device)
                print(f"✅ Loaded trained model from {model_path}")
                return _tryon_model
            except Exception as e:
                print(f"❌ Failed to load trained model: {e}")
                _tryon_model = None
        
        # If no trained model available, try to create an untrained one
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from models.tryon_model import VirtualTryOnModel
            _tryon_model = VirtualTryOnModel(device=device)
            print(f"⚠️ Using untrained model (no pre-trained weights)")
            return _tryon_model
        except Exception as e:
            print(f"❌ Failed to create model: {e}")
            _tryon_model = None
    
    return _tryon_model

def baseline_fit(person_image_path, cloth_image_path, pose_json_path=None, cloth_mask_path=None):
    """Fit clothing onto a person using the trained model or fallback to baseline warping."""
    
    # Determine request ID from the path for proper file organization
    request_id = None
    try:
        # Extract request ID from the path structure
        path_parts = Path(person_image_path).parts
        for i, part in enumerate(path_parts):
            if part == 'outputs' and i + 1 < len(path_parts):
                request_id = path_parts[i + 1]
                break
    except:
        pass
    
    # Set up result path
    if request_id:
        result_dir = os.path.join(settings.MEDIA_ROOT, 'outputs', str(request_id), 'generated')
        result_filename = f'tryon_result_{request_id}.jpg'
    else:
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        result_filename = 'model_result.jpg'
    
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, result_filename)
    
    try:
        # Try to use the trained model
        model = _get_tryon_model()
        if model and hasattr(model, 'try_on'):
            print("Using trained model for try-on...")
            result_img = model.try_on(person_image_path, cloth_image_path, pose_json_path, cloth_mask_path)
            
            # Ensure we have a valid result
            if result_img and isinstance(result_img, Image.Image):
                # Save the result with high quality
                result_img.save(result_path, 'JPEG', quality=95, optimize=True)
                print(f"✅ Model result saved to: {result_path}")
                return result_path
            else:
                raise ValueError("Model returned invalid result")
        else:
            raise AttributeError("Model does not have try_on method")
            
    except Exception as e:
        print(f"Model inference failed: {e}. Falling back to enhanced warping.")
        
        # Fallback to enhanced baseline warping
        try:
            person_img = Image.open(person_image_path).convert('RGB')
            cloth_img = Image.open(cloth_image_path).convert('RGB')
            
            # Ensure consistent sizing
            target_size = (512, 512)
            person_img = person_img.resize(target_size, Image.LANCZOS)
            cloth_img = cloth_img.resize(target_size, Image.LANCZOS)
            
            # Load pose keypoints if available
            if pose_json_path and os.path.exists(pose_json_path):
                with open(pose_json_path, 'r') as f:
                    pose_data = json.load(f)
                keypoints = pose_data.get("keypoints", [])
                
                if keypoints:
                    print("Warping cloth using pose data...")
                    # Warp cloth using enhanced warping
                    warped_cloth = enhanced_warping_cloth(cloth_img, keypoints, target_size)
                    
                    # Create a better blend using the warped cloth
                    result_img = create_better_blend(person_img, warped_cloth, cloth_mask_path)
                else:
                    print("No valid keypoints found, using intelligent overlay")
                    result_img = intelligent_overlay(person_img, cloth_img, cloth_mask_path)
            else:
                print("No pose data available, using intelligent overlay")
                result_img = intelligent_overlay(person_img, cloth_img, cloth_mask_path)
            
            # Save enhanced result
            fallback_filename = f'enhanced_result_{request_id}.jpg' if request_id else 'enhanced_result.jpg'
            fallback_path = os.path.join(result_dir, fallback_filename)
            result_img.save(fallback_path, 'JPEG', quality=95, optimize=True)
            print(f"✅ Enhanced warping result saved to: {fallback_path}")
            return fallback_path
            
        except Exception as fallback_error:
            print(f"Enhanced warping also failed: {fallback_error}")
            # Final fallback - intelligent blend
            try:
                person_img = Image.open(person_image_path).convert('RGB')
                cloth_img = Image.open(cloth_image_path).convert('RGB')
                
                # Create an intelligent blend with better positioning
                result_img = create_intelligent_blend(person_img, cloth_img, cloth_mask_path)
                
                final_filename = f'final_result_{request_id}.jpg' if request_id else 'final_result.jpg'
                final_path = os.path.join(result_dir, final_filename)
                result_img.save(final_path, 'JPEG', quality=90)
                print(f"✅ Final fallback result saved to: {final_path}")
                return final_path
                
            except Exception as final_error:
                print(f"Final fallback failed: {final_error}")
                raise

def enhanced_warping_cloth(cloth_img, keypoints, target_size):
    """Enhanced cloth warping with better keypoint handling."""
    cloth_array = np.array(cloth_img)
    height, width = target_size
    
    # Define cloth corners
    cloth_corners = np.array([
        [0, 0],           # top-left
        [width, 0],       # top-right
        [width, height],  # bottom-right
        [0, height]       # bottom-left
    ], dtype=np.float32)
    
    # Extract body landmarks with confidence checking
    def get_keypoint_safe(idx, default_x=width//2, default_y=height//2):
        if len(keypoints) > idx and keypoints[idx] is not None:
            if isinstance(keypoints[idx], (list, tuple)) and len(keypoints[idx]) >= 3:
                x, y, conf = keypoints[idx][:3]
                if conf is not None and conf >= 0.1:
                    return [max(0, min(width-1, int(x))), max(0, min(height-1, int(y)))]
        return [default_x, default_y]
    
    # Map to correct OpenPose COCO indices
    shoulder_right = get_keypoint_safe(2, width//3, height//3)    # Right shoulder
    shoulder_left = get_keypoint_safe(5, 2*width//3, height//3)   # Left shoulder  
    hip_right = get_keypoint_safe(8, width//3, 2*height//3)       # Right hip
    hip_left = get_keypoint_safe(11, 2*width//3, 2*height//3)     # Left hip
    
    # Define target points on the person (corrected order)
    body_corners = np.array([
        shoulder_left,   # top-left -> left shoulder
        shoulder_right,  # top-right -> right shoulder
        hip_right,       # bottom-right -> right hip
        hip_left         # bottom-left -> left hip
    ], dtype=np.float32)
    
    try:
        # Calculate perspective transform and warp
        M = cv2.getPerspectiveTransform(cloth_corners, body_corners)
        warped = cv2.warpPerspective(cloth_array, M, target_size)
        return Image.fromarray(warped)
    except Exception as e:
        print(f"Perspective transform failed: {e}, returning resized cloth")
        return cloth_img.resize(target_size)

def create_better_blend(person_img, warped_cloth, cloth_mask_path=None):
    """Create a better blend using masks and alpha compositing."""
    try:
        # Convert to numpy arrays
        person_array = np.array(person_img)
        cloth_array = np.array(warped_cloth)
        
        # Load or create cloth mask
        if cloth_mask_path and os.path.exists(cloth_mask_path):
            cloth_mask = Image.open(cloth_mask_path).convert('L').resize(person_img.size)
            mask_array = np.array(cloth_mask) / 255.0
        else:
            # Create a simple mask for the cloth region
            mask_array = np.ones((person_img.size[1], person_img.size[0])) * 0.7
        
        # Expand mask to 3 channels
        mask_3d = np.stack([mask_array] * 3, axis=-1)
        
        # Blend the images
        blended = (cloth_array * mask_3d + person_array * (1 - mask_3d)).astype(np.uint8)
        
        return Image.fromarray(blended)
        
    except Exception as e:
        print(f"Better blending failed: {e}, using simple blend")
        return Image.blend(person_img, warped_cloth, 0.6)

def intelligent_overlay(person_img, cloth_img, cloth_mask_path=None):
    """Intelligent overlay method with better positioning."""
    try:
        # Resize cloth to match person image
        cloth_resized = cloth_img.resize(person_img.size)
        
        # Load or create cloth mask
        if cloth_mask_path and os.path.exists(cloth_mask_path):
            cloth_mask = Image.open(cloth_mask_path).convert('L').resize(person_img.size)
            mask_array = np.array(cloth_mask) / 255.0
        else:
            # Create a torso-focused mask
            mask = Image.new('L', person_img.size, 0)
            draw = ImageDraw.Draw(mask)
            w, h = person_img.size
            # Focus on torso area
            draw.rectangle([w//4, h//3, 3*w//4, 2*h//3], fill=255)
            draw.ellipse([w//3, h//4, 2*w//3, h//2], fill=255)
            mask_array = np.array(mask) / 255.0
        
        # Apply Gaussian blur to mask for smoother blending
        mask_blurred = cv2.GaussianBlur(mask_array, (21, 21), 0)
        mask_3d = np.stack([mask_blurred] * 3, axis=-1)
        
        # Blend images
        person_array = np.array(person_img)
        cloth_array = np.array(cloth_resized)
        
        blended = (cloth_array * mask_3d + person_array * (1 - mask_3d)).astype(np.uint8)
        
        return Image.fromarray(blended)
        
    except Exception as e:
        print(f"Intelligent overlay failed: {e}")
        return simple_overlay(person_img, cloth_img)

def create_intelligent_blend(person_img, cloth_img, cloth_mask_path=None):
    """Create an intelligent blend with advanced techniques."""
    try:
        # Resize images to consistent size
        target_size = (512, 512)
        person_resized = person_img.resize(target_size, Image.LANCZOS)
        cloth_resized = cloth_img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy arrays
        person_array = np.array(person_resized)
        cloth_array = np.array(cloth_resized)
        
        # Create a sophisticated mask
        if cloth_mask_path and os.path.exists(cloth_mask_path):
            cloth_mask = Image.open(cloth_mask_path).convert('L').resize(target_size)
            mask_array = np.array(cloth_mask) / 255.0
        else:
            # Create body-aware mask
            mask = Image.new('L', target_size, 0)
            draw = ImageDraw.Draw(mask)
            w, h = target_size
            
            # Create a more realistic body shape mask
            # Torso area
            draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=180)
            # Upper torso (chest area)
            draw.ellipse([w//3, h//4, 2*w//3, h//2], fill=255)
            # Lower torso
            draw.rectangle([w//3, h//2, 2*w//3, 2*h//3], fill=200)
            
            mask_array = np.array(mask) / 255.0
        
        # Apply multiple blur levels for natural blending
        mask_fine = cv2.GaussianBlur(mask_array, (5, 5), 0)
        mask_coarse = cv2.GaussianBlur(mask_array, (25, 25), 0)
        
        # Combine masks for edge smoothing
        mask_combined = 0.7 * mask_fine + 0.3 * mask_coarse
        mask_3d = np.stack([mask_combined] * 3, axis=-1)
        
        # Apply color correction to match lighting
        cloth_corrected = adjust_cloth_lighting(cloth_array, person_array, mask_3d)
        
        # Blend with edge preservation
        result = (cloth_corrected * mask_3d + person_array * (1 - mask_3d)).astype(np.uint8)
        
        # Post-process for better integration
        result = enhance_integration(result, person_array, mask_3d)
        
        return Image.fromarray(result)
        
    except Exception as e:
        print(f"Intelligent blend failed: {e}")
        return Image.blend(person_img.resize((512, 512)), cloth_img.resize((512, 512)), 0.4)

def adjust_cloth_lighting(cloth_array, person_array, mask_3d):
    """Adjust cloth lighting to match person's lighting conditions."""
    try:
        # Calculate average brightness in masked regions
        person_masked = person_array * mask_3d
        cloth_masked = cloth_array * mask_3d
        
        person_brightness = np.mean(person_masked[mask_3d[:,:,0] > 0.1])
        cloth_brightness = np.mean(cloth_masked[mask_3d[:,:,0] > 0.1])
        
        if cloth_brightness > 0:
            # Adjust cloth brightness to match person
            brightness_ratio = person_brightness / cloth_brightness
            brightness_ratio = np.clip(brightness_ratio, 0.7, 1.3)  # Limit adjustment
            
            cloth_adjusted = cloth_array * brightness_ratio
            cloth_adjusted = np.clip(cloth_adjusted, 0, 255)
            
            return cloth_adjusted.astype(np.uint8)
        
        return cloth_array
        
    except Exception as e:
        print(f"Lighting adjustment failed: {e}")
        return cloth_array

def enhance_integration(result, person_array, mask_3d):
    """Enhance the integration between cloth and person."""
    try:
        # Sharpen the result slightly
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(result, -1, kernel)
        
        # Blend original and sharpened based on mask confidence
        mask_confidence = mask_3d[:,:,0]
        enhancement_factor = np.where(mask_confidence > 0.5, 0.3, 0.0)
        enhancement_factor = np.stack([enhancement_factor] * 3, axis=-1)
        
        enhanced = result * (1 - enhancement_factor) + sharpened * enhancement_factor
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
        
    except Exception as e:
        print(f"Integration enhancement failed: {e}")
        return result

def simple_overlay(person_img, cloth_img):
    """Simple overlay method as fallback."""
    try:
        # Resize cloth to match person image
        cloth_resized = cloth_img.resize(person_img.size)
        
        # Create a simple alpha blend
        result = Image.blend(person_img, cloth_resized, 0.5)
        return result
        
    except Exception as e:
        print(f"Simple overlay failed: {e}")
        return person_img
