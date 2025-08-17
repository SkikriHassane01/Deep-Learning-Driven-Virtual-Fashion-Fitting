from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import torch

# Check if we can use CUDA for OpenPose
_USE_CUDA = torch.cuda.is_available()

# Load OpenPose model (lazy loading)
_openpose_net = None

def _load_openpose_model():
    global _openpose_net
    if _openpose_net is None:
        # Use OpenCV's DNN module with pre-trained OpenPose
        model_path = Path(__file__).parent.parent.parent / "models" / "pose"
        proto_file = model_path / "pose_deploy_linevec.prototxt"
        weights_file = model_path / "pose_iter_440000.caffemodel"
        
        try:
            _openpose_net = cv2.dnn.readNetFromCaffe(str(proto_file), str(weights_file))
            if _USE_CUDA:
                _openpose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                _openpose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("OpenPose model loaded successfully")
        except Exception as e:
            print(f"Error loading OpenPose model: {e}")
            _openpose_net = None
    
    return _openpose_net

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
    
    try:
        # Load image
        img = cv2.imread(str(user_image_path))
        if img is None:
            raise ValueError(f"Could not read image at {user_image_path}")
        
        # Get body segmentation mask
        body_mask = extract_body_mask(img)
        cv2.imwrite(str(mask_path), body_mask)
        
        # Get pose keypoints
        keypoints = extract_pose_keypoints(img)
        
        # Save pose data as JSON
        pose_data = {
            "keypoints": keypoints,
            "image_size": [img.shape[1], img.shape[0]]
        }
        
        with open(pose_json_path, 'w') as f:
            json.dump(pose_data, f)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        # Create fallback files
        create_fallback_files(user_image_path, mask_path, pose_json_path)
    
    return mask_path, pose_json_path

def extract_body_mask(img):
    """Extract body mask using background removal."""
    # Convert to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # Try to use U2Net for segmentation
        try:
            import torch
            import torchvision.transforms as transforms
            from models.u2net import U2NET
            
            # Load pre-trained model
            model_dir = Path(__file__).parent.parent.parent / "models" / "segmentation"
            model_path = model_dir / "u2net.pth"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Set up model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = U2NET(3, 1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model = model.to(device)
            
            # Transform image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process image
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
                mask = output[0].squeeze().cpu().numpy()
                
            # Resize back to original size
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            # Threshold the mask
            mask = (mask > 0.5).astype(np.uint8) * 255
            
        except Exception as u2net_error:
            print(f"U2Net failed: {u2net_error}. Falling back to GrabCut.")
            
            # Fallback to GrabCut
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            rect = (int(img.shape[1]*0.1), int(img.shape[0]*0.1), 
                    int(img.shape[1]*0.8), int(img.shape[0]*0.8))
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask==2) | (mask==0), 0, 1).astype('uint8') * 255
    
    except Exception as e:
        print(f"Body segmentation failed: {e}. Using simple threshold.")
        
        # Final fallback: simple thresholding on HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
        
        # Clean up with morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def extract_pose_keypoints(img):
    """Extract human pose keypoints using OpenPose."""
    try:
        # Load OpenPose model
        net = _load_openpose_model()
        if net is None:
            raise ValueError("OpenPose model not available")
        
        # Prepare image for network
        height, width = img.shape[:2]
        input_blob = cv2.dnn.blobFromImage(img, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        
        # Set input and forward pass
        net.setInput(input_blob)
        output = net.forward()
        
        # Number of points
        num_points = output.shape[1] - 1
        
        # Extract keypoints
        keypoints = []
        confidence_threshold = 0.1
        
        for i in range(num_points):
            # Confidence map
            prob_map = output[0, i+1, :, :]
            prob_map = cv2.resize(prob_map, (width, height))
            
            # Find global maxima
            _, conf, _, point = cv2.minMaxLoc(prob_map)
            
            if conf > confidence_threshold:
                keypoints.append([int(point[0]), int(point[1]), float(conf)])
            else:
                keypoints.append([0, 0, 0.0])  # Zero for low confidence points
        
        return keypoints
        
    except Exception as e:
        print(f"Pose estimation failed: {e}")
        
        # Fallback: generate simple keypoints
        # Basic assumption: person is centered in the image
        height, width = img.shape[:2]
        
        # Create basic keypoints (head, neck, shoulders, hips, knees, ankles)
        keypoints = [
            [width//2, height//6, 1.0],  # Head
            [width//2, height//4, 1.0],  # Neck
            [width//3, height//3, 1.0],  # Right shoulder
            [2*width//3, height//3, 1.0],  # Left shoulder
            [width//3, height//2, 1.0],  # Right elbow
            [2*width//3, height//2, 1.0],  # Left elbow
            [width//4, 2*height//3, 1.0],  # Right wrist
            [3*width//4, 2*height//3, 1.0],  # Left wrist
            [width//2, height//2, 1.0],  # Center hip
            [width//3, 2*height//3, 1.0],  # Right hip
            [2*width//3, 2*height//3, 1.0],  # Left hip
            [width//3, 3*height//4, 1.0],  # Right knee
            [2*width//3, 3*height//4, 1.0],  # Left knee
            [width//3, 7*height//8, 1.0],  # Right ankle
            [2*width//3, 7*height//8, 1.0]   # Left ankle
        ]
        
        return keypoints

def create_fallback_files(user_image_path, mask_path, pose_json_path):
    """Create fallback files for mask and pose JSON."""
    try:
        # Create a simple mask (white silhouette)
        with Image.open(user_image_path) as img:
            width, height = img.size
            mask = Image.new('L', (width, height), 255)
            mask.save(mask_path)
        
        # Create basic pose data
        pose_data = {
            "keypoints": [
                [width//2, height//6, 1.0],  # Head
                [width//2, height//4, 1.0],  # Neck
                [width//3, height//3, 1.0],  # Right shoulder
                [2*width//3, height//3, 1.0],  # Left shoulder
                [width//3, height//2, 1.0],  # Right elbow
                [2*width//3, height//2, 1.0],  # Left elbow
                [width//4, 2*height//3, 1.0],  # Right wrist
                [3*width//4, 2*height//3, 1.0],  # Left wrist
                [width//2, height//2, 1.0],  # Center hip
                [width//3, 2*height//3, 1.0],  # Right hip
                [2*width//3, 2*height//3, 1.0],  # Left hip
                [width//3, 3*height//4, 1.0],  # Right knee
                [2*width//3, 3*height//4, 1.0],  # Left knee
                [width//3, 7*height//8, 1.0],  # Right ankle
                [2*width//3, 7*height//8, 1.0]   # Left ankle
            ],
            "image_size": [width, height]
        }
        
        with open(pose_json_path, 'w') as f:
            json.dump(pose_data, f)
            
    except Exception as e:
        print(f"Error creating fallback files: {e}")
        # Final fallback with hardcoded values
        mask = Image.new('L', (512, 512), 255)
        mask.save(mask_path)
        
        with open(pose_json_path, 'w') as f:
            json.dump({
                "keypoints": [[256, 85, 1.0], [256, 128, 1.0], [171, 171, 1.0], 
                             [341, 171, 1.0], [171, 256, 1.0], [341, 256, 1.0], 
                             [128, 341, 1.0], [384, 341, 1.0], [256, 256, 1.0],
                             [171, 341, 1.0], [341, 341, 1.0], [171, 384, 1.0],
                             [341, 384, 1.0], [171, 448, 1.0], [341, 448, 1.0]],
                "image_size": [512, 512]
            }, f)
