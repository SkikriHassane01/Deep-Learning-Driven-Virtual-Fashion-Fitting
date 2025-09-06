"""Image preprocess4g and postprocessing utilities."""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageProcessor:
    """Image processing utilities for the virtual fashion try-on system"""
    
    @staticmethod
    def resize_image(image: Union[np.ndarray, Image.Image], size: Tuple[int, int]) -> Union[np.ndarray, Image.Image]:
        """Resize image to target size"""
        if isinstance(image, Image.Image):
            return image.resize(size)
        else:
            return cv2.resize(image, size)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: list = [0.485, 0.456, 0.406], 
                       std: list = [0.229, 0.224, 0.225]) -> np.ndarray:
        """Normalize image with mean and std"""
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return image
    
    @staticmethod
    def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image tensor for visualization"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor.clone()
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if needed
        if tensor.min() < 0:
            tensor = ImageProcessor.denormalize_image(tensor)
        
        # Convert to numpy
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return image
    
    @staticmethod
    def image_to_tensor(image: Union[np.ndarray, Image.Image], 
                       normalize: bool = True, device: torch.device = None) -> torch.Tensor:
        """Convert image to tensor"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if normalize:
            tensor = tensor / 255.0
        
        if device:
            tensor = tensor.to(device)
        
        return tensor.unsqueeze(0)
    
    @staticmethod
    def create_mask_from_segmentation(segmentation: np.ndarray, 
                                    target_classes: list,
                                    blur_kernel: int = 21) -> np.ndarray:
        """Create smooth mask from segmentation for specific classes"""
        mask = np.zeros_like(segmentation, dtype=np.uint8)
        
        for cls in target_classes:
            mask[segmentation == cls] = 255
        
        # Apply Gaussian blur for smooth edges
        if blur_kernel > 0:
            mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        
        return mask
    
    @staticmethod
    def blend_images(original: Union[np.ndarray, Image.Image], 
                    generated: Union[np.ndarray, Image.Image],
                    mask: np.ndarray, alpha: float = None) -> Union[np.ndarray, Image.Image]:
        """Blend two images using a mask"""
        # Convert to numpy if needed
        return_pil = isinstance(original, Image.Image)
        
        if isinstance(original, Image.Image):
            original = np.array(original)
        if isinstance(generated, Image.Image):
            generated = np.array(generated)
        
        # Ensure same dimensions
        if original.shape != generated.shape:
            generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
        
        # Normalize mask
        mask_norm = mask.astype(np.float32) / 255.0
        if alpha is not None:
            mask_norm = mask_norm * alpha
        
        # Create 3D mask
        if len(mask_norm.shape) == 2:
            mask_3d = np.stack([mask_norm] * 3, axis=2)
        else:
            mask_3d = mask_norm
        
        # Blend images
        result = generated * mask_3d + original * (1 - mask_3d)
        result = result.astype(np.uint8)
        
        # Return in original format
        if return_pil:
            return Image.fromarray(result)
        return result
    
    @staticmethod
    def apply_color_to_mask(image: Union[np.ndarray, Image.Image],
                          mask: np.ndarray, color: Tuple[int, int, int]) -> Union[np.ndarray, Image.Image]:
        """Apply solid color to masked regions"""
        return_pil = isinstance(image, Image.Image)
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Apply color to masked area
        mask_bool = mask > 128
        image_np[mask_bool] = color
        
        if return_pil:
            return Image.fromarray(image_np)
        return image_np
    
    @staticmethod
    def extract_color_from_text(text: str) -> Tuple[int, int, int]:
        """Extract color from text description"""
        color_map = {
            'red': (200, 50, 50),
            'blue': (50, 50, 200), 
            'green': (50, 200, 50),
            'black': (50, 50, 50),
            'white': (230, 230, 230),
            'yellow': (200, 200, 50),
            'purple': (150, 50, 150),
            'pink': (255, 150, 150),
            'brown': (139, 69, 19),
            'orange': (255, 140, 0),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128)
        }
        
        text_lower = text.lower()
        for color_name, color_value in color_map.items():
            if color_name in text_lower:
                return color_value
        
        # Default color if none found
        return (100, 100, 150)


def preprocess_image(image: Image.Image, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        size: Target size (width, height)
        device: Device to move tensor to
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to numpy and resize
    image_np = np.array(image.resize(size))
    
    # Normalize
    image_np = image_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    # Convert to tensor and move to device
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().to(device)
    
    return tensor


def postprocess_prediction(prediction: torch.Tensor) -> np.ndarray:
    """
    Postprocess model prediction.
    
    Args:
        prediction: Model prediction tensor
        
    Returns:
        Segmentation mask as numpy array
    """
    if isinstance(prediction, tuple):
        prediction = prediction[-1]  # Use the last output (refined)
    
    # Get class predictions
    pred_mask = prediction.argmax(dim=0).cpu().numpy()
    
    return pred_mask.astype(np.uint8)