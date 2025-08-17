from __future__ import annotations
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
from .auto_mask import auto_cloth_mask
from PIL import Image
import random

# Load once (module-level) for performance
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DT = torch.float16 if _DEVICE == "cuda" else torch.float32

# A good, widely-used base; you can swap to SDXL later
_MODEL_ID = "runwayml/stable-diffusion-v1-5"

try:
    _pipe = StableDiffusionPipeline.from_pretrained(
        _MODEL_ID,
        torch_dtype=_DT,
        safety_checker=None,
        use_safetensors=True
    ).to(_DEVICE)
    print(f"Stable Diffusion loaded on {_DEVICE}")
except Exception as e:
    print(f"Warning: Could not load Stable Diffusion: {e}")
    _pipe = None

def generate_cloth_from_prompt(prompt: str, out_dir: Path, seed: int | None = 42) -> tuple[Path, Path]:
    """
    Generate a 512x512 cloth image from text.
    Returns (cloth_image_path, cloth_mask_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if _pipe is not None:
        # Use real Stable Diffusion
        full_prompt = (
            f"{prompt}. studio product shot, flat lay or frontal view, "
            "plain white background, high detail fabric, centered, clothing item"
        )
        negative = "people, hands, body, busy background, low quality, blurry, person wearing"

        generator = torch.Generator(device=_DEVICE)
        if seed is not None:
            generator = generator.manual_seed(seed)

        img: Image.Image = _pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            height=512, width=512,
            guidance_scale=7.5,
            num_inference_steps=20,  # Faster generation
            generator=generator
        ).images[0]

        cloth_path = out_dir / "gen_cloth.png"
        img.save(cloth_path)

        # Auto-create mask using GrabCut
        mask_path = out_dir / "gen_cloth_mask.png"
        auto_cloth_mask(cloth_path, mask_path)
    else:
        # Fallback to simple generation
        cloth_path, mask_path = generate_cloth_from_prompt_v2(prompt, out_dir)

    return cloth_path, mask_path

def generate_cloth_from_prompt_v2(prompt, out_dir):
    """
    Fallback cloth generation using simple graphics.
    """
    cloth_img_path = out_dir / "generated_cloth.png"
    cloth_mask_path = out_dir / "generated_cloth_mask.png"
    
    try:
        size = (512, 512)
        
        # Generate color based on prompt
        colors = {
            'red': (220, 20, 60),
            'blue': (30, 144, 255),
            'green': (34, 139, 34),
            'black': (25, 25, 25),
            'white': (248, 248, 255),
            'yellow': (255, 215, 0),
            'pink': (255, 20, 147),
            'purple': (138, 43, 226),
            'orange': (255, 140, 0),
            'brown': (139, 69, 19),
        }
        
        # Choose color based on prompt keywords
        cloth_color = (100, 100, 100)  # default gray
        for color_name, color_rgb in colors.items():
            if color_name in prompt.lower():
                cloth_color = color_rgb
                break
        else:
            # Random bright color if no specific color mentioned
            cloth_color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        
        # Create cloth image with a clothing-like shape
        cloth_img = Image.new('RGB', size, (255, 255, 255))  # White background
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(cloth_img)
        
        # Draw a t-shirt like shape
        # Main body
        draw.rectangle([150, 200, 362, 450], fill=cloth_color)
        # Sleeves
        draw.rectangle([100, 200, 150, 300], fill=cloth_color)
        draw.rectangle([362, 200, 412, 300], fill=cloth_color)
        # Neck area
        draw.rectangle([220, 150, 292, 200], fill=cloth_color)
        
        # Add some texture or pattern
        if 'stripe' in prompt.lower():
            for y in range(200, 450, 20):
                draw.rectangle([150, y, 362, y+10], fill=tuple(max(0, c-30) for c in cloth_color))
        
        cloth_img.save(cloth_img_path)
        
        # Create corresponding mask
        mask = Image.new('L', size, 0)  # Black background
        mask_draw = ImageDraw.Draw(mask)
        
        # Same shape but white for cloth area
        mask_draw.rectangle([150, 200, 362, 450], fill=255)
        mask_draw.rectangle([100, 200, 150, 300], fill=255)
        mask_draw.rectangle([362, 200, 412, 300], fill=255)
        mask_draw.rectangle([220, 150, 292, 200], fill=255)
        
        mask.save(cloth_mask_path)
        
    except Exception as e:
        print(f"Error in fallback generation: {e}")
        # Final fallback
        Image.new('RGB', (512, 512), (150, 150, 150)).save(cloth_img_path)
        Image.new('L', (512, 512), 255).save(cloth_mask_path)
    
    return cloth_img_path, cloth_mask_path
