"""Production inference interface for the virtual fashion try-on system."""

import os
import json
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Optional, Union

from ..models.human_parser import HumanParsingNet
from ..utils.config import Config


class Predictor:
    """Inference class for Human Parsing model and Virtual Try-On system"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to model configuration file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # Initialize model
        self.model = HumanParsingNet(num_classes=self.config['model']['num_classes'])
        
        # Load weights
        self._load_model_weights(model_path)
        
        # Setup preprocessing
        self.transform = A.Compose([
            A.Resize(self.config['preprocessing']['size'][0],
                    self.config['preprocessing']['size'][1]),
            A.Normalize(mean=self.config['preprocessing']['mean'],
                       std=self.config['preprocessing']['std']),
            ToTensorV2()
        ])
        
        print(f"Predictor initialized on {self.device}")
    
    def _default_config(self) -> Dict:
        """Default configuration for inference"""
        return {
            'model': {
                'num_classes': 18,
                'input_size': [512, 512]
            },
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'size': [512, 512]
            }
        }
    
    def _load_model_weights(self, model_path: str):
        """Load model weights from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded model weights from {model_path}")
            
        except Exception as e:
            print(f"Warning: Failed to load model weights: {e}")
            print("Using randomly initialized weights")
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Predict segmentation mask for an image.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Segmentation mask as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Store original size for resizing back
        original_size = image.shape[:2]
        
        # Preprocess
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.argmax(dim=1)[0].cpu().numpy()
        
        # Resize back to original size if needed
        if original_size != tuple(self.config['preprocessing']['size']):
            import cv2
            prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        return prediction
    
    def predict_from_path(self, image_path: str) -> np.ndarray:
        """
        Load image from path and predict.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Segmentation mask as numpy array
        """
        image = np.array(Image.open(image_path).convert('RGB'))
        return self.predict(image)
    
    def predict_batch(self, images: list) -> list:
        """
        Predict on a batch of images.
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            
        Returns:
            List of segmentation masks
        """
        predictions = []
        for image in images:
            prediction = self.predict(image)
            predictions.append(prediction)
        return predictions
    
    def get_class_predictions(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Get detailed class predictions with confidence scores.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with class predictions and statistics
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Preprocess
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Get predictions with probabilities
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)[0].cpu().numpy()
            probs = probabilities[0].cpu().numpy()
        
        # Compute class statistics
        unique_classes = np.unique(predictions)
        class_stats = {}
        
        for class_id in unique_classes:
            mask = predictions == class_id
            pixel_count = mask.sum()
            avg_confidence = probs[class_id][mask].mean() if mask.sum() > 0 else 0.0
            percentage = (pixel_count / predictions.size) * 100
            
            class_stats[int(class_id)] = {
                'pixel_count': int(pixel_count),
                'percentage': float(percentage),
                'avg_confidence': float(avg_confidence)
            }
        
        return {
            'predictions': predictions,
            'class_statistics': class_stats,
            'total_pixels': int(predictions.size)
        }
    
    def visualize_prediction(self, image: Union[np.ndarray, Image.Image], 
                           save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize prediction with colored overlay.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as PIL Image
        """
        from ..utils.visualization import Visualizer
        from ..utils.image_processing import ImageProcessor
        
        # Get prediction
        prediction = self.predict(image)
        
        # Convert to colored mask
        colored_mask = Visualizer.mask_to_color(prediction)
        
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Blend with original image
        result = ImageProcessor.blend_images(image_np, colored_mask, 
                                           np.ones_like(prediction) * 128, 
                                           alpha=0.5)
        
        # Convert to PIL
        result_pil = Image.fromarray(result) if isinstance(result, np.ndarray) else result
        
        # Save if requested
        if save_path:
            result_pil.save(save_path)
            print(f"Visualization saved to {save_path}")
        
        return result_pil