"""Simple garment generator using Stable Diffusion."""

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings('ignore')


class SimpleGarmentGenerator:
    """Simple garment generator using Stable Diffusion"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = None):
        """
        Initialize the garment generator.
        
        Args:
            model_name: HuggingFace model name for Stable Diffusion
            device: Device to run on ('cuda' or 'cpu'). Auto-detect if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"🔄 Loading Stable Diffusion model: {model_name}")
        
        # Load the pre-trained model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory efficient settings
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print(f"✅ Model loaded successfully on {self.device}!")
    
    def generate_outfit(self, prompt: str, seed: int = 42, image_size: int = 512,
                       num_inference_steps: int = 20, guidance_scale: float = 7.5) -> Image.Image:
        """
        Generate outfit from text prompt.
        
        Args:
            prompt: Text description of the outfit
            seed: Random seed for reproducible results
            image_size: Output image size (width and height)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            
        Returns:
            PIL Image of the generated outfit
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Enhanced prompt for better fashion results
        fashion_prompt = f"high quality fashion photography, {prompt}, professional lighting, detailed fabric texture, fashion model, studio photography"
        
        # Negative prompt to avoid bad results
        negative_prompt = "low quality, blurry, distorted, ugly, bad anatomy, extra limbs"
        
        print(f"🎨 Generating: {prompt}")
        
        # Generate image
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=fashion_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=image_size,
                width=image_size
            ).images[0]
        
        return result
    
    def generate_batch(self, prompts: list, **kwargs) -> list:
        """
        Generate multiple outfits from a list of prompts.
        
        Args:
            prompts: List of text descriptions
            **kwargs: Additional arguments passed to generate_outfit
            
        Returns:
            List of PIL Images
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Generating {i+1}/{len(prompts)}: {prompt}")
            result = self.generate_outfit(prompt, **kwargs)
            results.append(result)
        return results