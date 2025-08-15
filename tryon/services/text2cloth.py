from __future__ import annotations
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
from .auto_mask import auto_cloth_mask  # we wrote this earlier (GrabCut)
from PIL import Image, ImageDraw, ImageFont
import random

# Load once (module-level) for performance
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DT = torch.float16 if _DEVICE == "cuda" else torch.float32

# A good, widely-used base; you can swap to SDXL later
_MODEL_ID = "runwayml/stable-diffusion-v1-5"

_pipe = StableDiffusionPipeline.from_pretrained(
    _MODEL_ID,
    torch_dtype=_DT,
    safety_checker=None,  # keep simple; add your own moderation if needed
    use_safetensors=True
).to(_DEVICE)

def generate_cloth_from_prompt(prompt: str, out_dir: Path, seed: int | None = 42) -> tuple[Path, Path]:
    """
    Generate a 512x512 cloth image from text.
    Returns (cloth_image_path, cloth_mask_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guide the model to make a clean product photo
    full_prompt = (
        f"{prompt}. studio product shot, flat lay or frontal view, "
        "plain white background, high detail fabric, centered"
    )
    negative = "people, hands, body, busy background, low quality, blurry"

    generator = torch.Generator(device=_DEVICE)
    if seed is not None:
        generator = generator.manual_seed(seed)

    img: Image.Image = _pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        height=512, width=512,
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator
    ).images[0]

    cloth_path = out_dir / "gen_cloth.png"
    img.save(cloth_path)

    # Auto-create mask (PNG, 0/255) using GrabCut or alpha if present
    mask_path = out_dir / "gen_cloth_mask.png"
    auto_cloth_mask(cloth_path, mask_path)

    return cloth_path, mask_path

def generate_cloth_from_prompt_v2(prompt, out_dir):
    """
    Generate cloth image and mask from text prompt.
    
    Args:
        prompt: Text description of the desired cloth
        out_dir: Output directory for generated files
        
    Returns:
        tuple: (cloth_img_path, cloth_mask_path)
    """
    cloth_img_path = out_dir / "generated_cloth.png"
    cloth_mask_path = out_dir / "generated_cloth_mask.png"
    
    try:
        # Create a simple colored rectangle as placeholder cloth
        size = (512, 512)
        
        # Generate random color based on prompt
        colors = {
            'red': (255, 100, 100),
            'blue': (100, 100, 255),
            'green': (100, 255, 100),
            'black': (50, 50, 50),
            'white': (245, 245, 245),
            'yellow': (255, 255, 100),
            'pink': (255, 150, 200),
        }
        
        # Choose color based on prompt keywords
        cloth_color = (150, 150, 150)  # default gray
        for color_name, color_rgb in colors.items():
            if color_name in prompt.lower():
                cloth_color = color_rgb
                break
        else:
            # Random color if no specific color mentioned
            cloth_color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
        
        # Create cloth image
        cloth_img = Image.new('RGB', size, cloth_color)
        draw = ImageDraw.Draw(cloth_img)
        
        # Add simple pattern/text
        try:
            # Try to add text with prompt
            font_size = 24
            draw.text((50, size[1]//2), f"Generated: {prompt[:20]}...", 
                     fill=(255, 255, 255) if sum(cloth_color) < 400 else (0, 0, 0))
        except:
            pass
        
        cloth_img.save(cloth_img_path)
        
        # Create mask (white cloth area, black background)
        mask = Image.new('L', size, 0)
        mask_draw = ImageDraw.Draw(mask)
        # Create a shirt-like shape
        mask_draw.rectangle([100, 150, 412, 450], fill=255)
        mask.save(cloth_mask_path)
        
    except Exception as e:
        print(f"Error generating cloth: {e}")
        # Fallback: create simple images
        Image.new('RGB', (512, 512), (150, 150, 150)).save(cloth_img_path)
        Image.new('L', (512, 512), 255).save(cloth_mask_path)
    
    return cloth_img_path, cloth_mask_path
