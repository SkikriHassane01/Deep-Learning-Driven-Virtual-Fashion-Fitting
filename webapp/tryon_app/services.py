"""
Simple service to integrate with AI models.
"""

import os
import sys
import logging
from PIL import Image
from pathlib import Path
from django.conf import settings

# Set up logging
logger = logging.getLogger('tryon_app')

# Add src directory to Python path to import our models
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    # Check for required dependencies first
    import torch
    import cv2
    import numpy as np
    logger.info("Core dependencies (torch, cv2, numpy) available")
    
    # Try to import diffusers
    try:
        import diffusers
        logger.info("Diffusers library available - full AI pipeline possible")
        diffusers_available = True
    except ImportError:
        logger.warning("Diffusers library not available - will use fallback generation")
        diffusers_available = False
    
    # Import our models
    from src.models.virtual_tryon import VirtualTryOnPipeline
    from src.models.human_parser import HumanParsingNet
    logger.info("Successfully imported AI models")
    
except ImportError as e:
    logger.error(f"Critical dependencies missing: {e}")
    logger.error("Please install required packages: torch, torchvision, opencv-python, numpy")
    VirtualTryOnPipeline = None


class VirtualTryOnService:
    """Simple service to handle virtual try-on processing"""
    
    def __init__(self):
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the AI pipeline with detailed error handling"""
        try:
            if VirtualTryOnPipeline is None:
                logger.error("VirtualTryOnPipeline class not available - check dependencies")
                return
                
            # Use Django settings for model path
            model_path = getattr(settings, 'HUMAN_PARSER_MODEL_PATH', None)
            if not model_path:
                # Fallback to default path
                model_path = PROJECT_ROOT / "model" / "human_parser_model.pth"
            
            logger.info(f"Initializing pipeline with model path: {model_path}")
            logger.info(f"Model file exists: {Path(model_path).exists()}")
            
            if Path(model_path).exists():
                logger.info(f"Loading trained model from: {model_path}")
                self.pipeline = VirtualTryOnPipeline(human_parser_path=str(model_path))
                logger.info("✓ AI pipeline initialized successfully with trained model")
            else:
                logger.warning(f"Model file not found at {model_path}")
                logger.info("Attempting to initialize with untrained model...")
                self.pipeline = VirtualTryOnPipeline()
                logger.info("✓ AI pipeline initialized with untrained model")
                
        except Exception as e:
            logger.error(f"✗ Failed to initialize AI pipeline: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.pipeline = None
    
    def process_image(self, image_path, clothing_prompt):
        """
        Process image with virtual try-on.
        
        Args:
            image_path: Path to uploaded image
            clothing_prompt: Text description of desired clothing
            
        Returns:
            dict: Result containing success status and result image path or error
        """
        logger.info(f"Processing image: {image_path} with prompt: '{clothing_prompt}'")
        
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded image size: {image.size}")
            
            # Create results directory
            result_filename = f"result_{Path(image_path).stem}.png"
            result_path = Path(image_path).parent.parent / 'results' / result_filename
            result_path.parent.mkdir(exist_ok=True)
            logger.info(f"Result will be saved to: {result_path}")
            
            if self.pipeline:
                logger.info("Using AI pipeline for processing")
                
                # Use AI pipeline - this returns a dictionary with multiple components
                try_on_result = self.pipeline.try_on(image, clothing_prompt)
                logger.info("AI pipeline processing completed successfully")
                
                # Extract the final result image from the dictionary
                final_result_image = try_on_result['result']
                
                # Save the final result image
                final_result_image.save(result_path)
                logger.info(f"Result image saved to: {result_path}")
                
                return {
                    'success': True,
                    'result_path': result_path,
                    'message': f'Successfully generated virtual try-on for: {clothing_prompt}'
                }
            else:
                logger.warning("AI pipeline not available, using simple fallback")
                
                # Simple fallback - apply basic color transformation
                fallback_result = self._create_simple_fallback(image, clothing_prompt)
                fallback_result.save(result_path)
                
                return {
                    'success': True,
                    'result_path': result_path,
                    'message': f'Processed with simple transform (AI not available): {clothing_prompt}'
                }
                
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'result_path': None
            }
    
    def _create_simple_fallback(self, image, prompt):
        """Create a simple color-based fallback when AI is not available"""
        logger.info("Creating simple fallback transformation")
        
        import numpy as np
        result_np = np.array(image)
        
        # Simple color mapping based on prompt keywords
        color_map = {
            'red': (200, 50, 50), 'blue': (50, 50, 200), 'green': (50, 200, 50),
            'black': (50, 50, 50), 'white': (230, 230, 230), 'yellow': (200, 200, 50),
            'purple': (150, 50, 150), 'pink': (255, 150, 150), 'brown': (139, 69, 19),
            'denim': (70, 100, 150), 'shirt': (120, 150, 200), 'dress': (180, 120, 180)
        }
        
        # Find matching color from prompt
        color = (120, 150, 180)  # Default blue
        for keyword, color_value in color_map.items():
            if keyword.lower() in prompt.lower():
                color = color_value
                logger.info(f"Applied {keyword} color transformation")
                break
        
        # Apply color to clothing region (simple approximation)
        h, w = result_np.shape[:2]
        y_start, y_end = int(h * 0.25), int(h * 0.75)
        x_start, x_end = int(w * 0.25), int(w * 0.75)
        
        # Create a simple blend
        overlay = result_np.copy()
        overlay[y_start:y_end, x_start:x_end] = color
        
        # Blend with original (50% opacity)
        result_np = (result_np * 0.7 + overlay * 0.3).astype(np.uint8)
        
        return Image.fromarray(result_np)
    
    def is_available(self):
        """Check if the AI pipeline is available"""
        return self.pipeline is not None


# Global service instance
tryon_service = VirtualTryOnService()