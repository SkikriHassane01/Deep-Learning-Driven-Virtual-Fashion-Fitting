"""Virtual Try-On pipeline combining human parsing and garment generation."""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionInpaintPipeline
from typing import Dict, Optional, Tuple

from .human_parser import HumanParsingNet


class VirtualTryOnPipeline:
    """Complete virtual try-on pipeline"""
    
    def __init__(self, human_parser_path: str = None, device: str = None):
        """
        Initialize the virtual try-on pipeline.
        
        Args:
            human_parser_path: Path to trained human parsing model
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = (512, 512)
        self.clothing_classes = [4, 5, 7]  # Upper-clothes, Skirt, Dress
        
        print("VIRTUAL TRY-ON PIPELINE")
        print("=" * 50)
        print(f"Device: {self.device}")
        
        # Load Human Parsing Model
        self._load_human_parser(human_parser_path)
        
        # Load Diffusion Model
        self._load_diffusion_model()
        
        print("Pipeline initialized successfully!")
    
    def _load_human_parser(self, model_path: str = None):
        """Load trained Human Parsing Model"""
        print("Loading Human Parsing Model...")
        
        try:
            self.parser = HumanParsingNet(num_classes=18)
            
            if model_path and os.path.exists(model_path):
                print(f"Loading weights from: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load the state dict
                self.parser.load_state_dict(state_dict, strict=False)
                print("Successfully loaded trained weights!")
                self.parser_loaded = True
            else:
                if model_path:
                    print(f"Model file not found at {model_path}")
                print("Using initialized model without trained weights")
                self.parser_loaded = False
            
            self.parser.to(self.device)
            self.parser.eval()
            print("Human Parser ready for inference")
            
        except Exception as e:
            print(f"Failed to load Human Parsing Model: {e}")
            self.parser_loaded = False
    
    def _load_diffusion_model(self):
        """Load Stable Diffusion inpainting model"""
        print("Loading Diffusion Model...")
        
        try:
            self.diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            print("Diffusion model loaded successfully")
            self.diffusion_loaded = True
            
        except Exception as e:
            print(f"Failed to load diffusion model: {e}")
            print("Will use fallback color generation")
            self.diffusion_loaded = False
    
    def segment_human(self, image: Image.Image) -> np.ndarray:
        """Segment human body parts"""
        if not self.parser_loaded:
            return self._create_simple_mask(image)
        
        # Preprocess image
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, self.image_size)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.parser(image_tensor)
                segmentation = output.argmax(dim=1)[0].cpu().numpy()
            
            # Resize back to original size
            if image_np.shape[:2] != self.image_size:
                segmentation = cv2.resize(segmentation, (image_np.shape[1], image_np.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            
            return segmentation.astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: Model inference failed: {e}")
            return self._create_simple_mask(image)
    
    def _create_simple_mask(self, image: Image.Image) -> np.ndarray:
        """Create simple body mask when parsing fails"""
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create upper body region (class 4 = upper clothes)
        y_start, y_end = int(h * 0.2), int(h * 0.6)
        x_center = w // 2
        x_width = int(w * 0.3)
        mask[y_start:y_end, x_center-x_width:x_center+x_width] = 4
        
        return mask
    
    def create_clothing_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """Create mask for clothing regions"""
        mask = np.zeros_like(segmentation, dtype=np.uint8)
        for cls in self.clothing_classes:
            mask[segmentation == cls] = 255
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask
    
    def generate_clothing(self, image: Image.Image, mask: np.ndarray, prompt: str) -> Image.Image:
        """Generate new clothing"""
        if self.diffusion_loaded:
            try:
                enhanced_prompt = f"{prompt}, high quality, detailed clothing, fashion photography"
                
                # Resize inputs for diffusion model (typically expects 512x512)
                original_size = image.size
                diffusion_size = (512, 512)
                
                # Resize image and mask for diffusion
                image_resized = image.resize(diffusion_size, Image.Resampling.LANCZOS)
                mask_resized = cv2.resize(mask, diffusion_size, interpolation=cv2.INTER_NEAREST)
                mask_image = Image.fromarray(mask_resized)
                
                result = self.diffusion_pipe(
                    prompt=enhanced_prompt,
                    image=image_resized,
                    mask_image=mask_image,
                    num_inference_steps=25,
                    guidance_scale=8.0,
                ).images[0]
                
                # Resize result back to original dimensions
                if result.size != original_size:
                    result = result.resize(original_size, Image.Resampling.LANCZOS)
                    print(f"Resized diffusion result from {diffusion_size} to {original_size}")
                
                return result
                
            except Exception as e:
                print(f"Warning: Diffusion generation failed: {e}")
                return self._simple_color_generation(image, mask, prompt)
        else:
            return self._simple_color_generation(image, mask, prompt)
    
    def _simple_color_generation(self, image: Image.Image, mask: np.ndarray, prompt: str) -> Image.Image:
        """Fallback color-based generation"""
        result = image.copy()
        result_np = np.array(result)
        
        # Extract color from prompt
        color_map = {
            'red': (200, 50, 50), 'blue': (50, 50, 200), 'green': (50, 200, 50),
            'black': (50, 50, 50), 'white': (230, 230, 230), 'yellow': (200, 200, 50),
            'purple': (150, 50, 150), 'pink': (255, 150, 150), 'brown': (139, 69, 19)
        }
        
        color = (100, 100, 150)  # Default
        for color_name, color_value in color_map.items():
            if color_name.lower() in prompt.lower():
                color = color_value
                break
        
        # Apply color to masked area
        mask_bool = mask > 128
        result_np[mask_bool] = color
        
        return Image.fromarray(result_np.astype(np.uint8))
    
    def blend_images(self, original: Image.Image, generated: Image.Image, mask: np.ndarray) -> Image.Image:
        """Blend original and generated images with size consistency"""
        # Ensure all inputs have the same dimensions
        original_size = original.size
        
        # Resize generated image to match original if needed
        if generated.size != original_size:
            generated = generated.resize(original_size, Image.Resampling.LANCZOS)
            print(f"Resized generated image from {generated.size} to {original_size}")
        
        # Resize mask to match original dimensions if needed
        if mask.shape[:2] != (original_size[1], original_size[0]):  # mask is (height, width)
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
            print(f"Resized mask to {original_size}")
        
        # Convert to numpy arrays
        original_np = np.array(original)
        generated_np = np.array(generated)
        
        # Ensure all arrays have the same shape
        if original_np.shape != generated_np.shape:
            raise ValueError(f"Shape mismatch after resizing: original {original_np.shape}, generated {generated_np.shape}")
        
        # Normalize mask
        mask_norm = mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_norm] * 3, axis=2)
        
        # Ensure mask has same dimensions as images
        if mask_3d.shape != original_np.shape:
            raise ValueError(f"Mask shape {mask_3d.shape} doesn't match image shape {original_np.shape}")
        
        # Blend with smooth transition
        result = generated_np * mask_3d + original_np * (1 - mask_3d)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def try_on(self, image: Image.Image, prompt: str) -> Dict:
        """
        Perform virtual try-on.
        
        Args:
            image: Input person image
            prompt: Description of desired clothing
            
        Returns:
            Dictionary with results including segmentation, mask, generated image, and final result
        """
        print(f"Processing: '{prompt}'")
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 1: Segment human body
        print("  - Segmenting human body...")
        segmentation = self.segment_human(image)
        
        # Step 2: Create clothing mask
        print("  - Creating target area mask...")
        clothing_mask = self.create_clothing_mask(segmentation)
        
        # Step 3: Generate new clothing
        print("  - Generating new clothing...")
        generated_clothing = self.generate_clothing(image, clothing_mask, prompt)
        
        # Step 4: Blend final result
        print("  - Blending final result...")
        final_result = self.blend_images(image, generated_clothing, clothing_mask)
        
        return {
            'original': image,
            'segmentation': segmentation,
            'mask': clothing_mask,
            'generated': generated_clothing,
            'result': final_result,
            'prompt': prompt
        }
    
    def process_single_prompt(self, image: Image.Image, prompt: str) -> Dict:
        """
        Process a single prompt for virtual try-on (alias for try_on method).
        
        Args:
            image: Input person image
            prompt: Description of desired clothing
            
        Returns:
            Dictionary with results including segmentation, mask, generated image, and final result
        """
        return self.try_on(image, prompt)