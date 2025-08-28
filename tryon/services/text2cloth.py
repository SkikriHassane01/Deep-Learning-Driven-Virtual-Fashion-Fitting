from __future__ import annotations
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch
import random
from .auto_mask import auto_cloth_mask
from PIL import Image, ImageDraw
import os

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
    cloth_path = out_dir / "gen_cloth.png"
    mask_path = out_dir / "gen_cloth_mask.png"
    
    # Print debugging info
    print(f"Generating cloth from prompt: '{prompt}'")
    print(f"Device for generation: {_DEVICE}")
    print(f"Output directory: {out_dir}")
    
    if _pipe is not None:
        try:
            # Use real Stable Diffusion with more specific prompts
            full_prompt = (
                f"A {prompt}, clothing item, fashion product photography, "
                "studio lighting, white background, high quality, detailed fabric texture, "
                "frontal view, flat lay style, professional product shot"
            )
            negative = (
                "person, human, body, hands, face, wearing, model, mannequin, "
                "low quality, blurry, dark background, multiple items, text, watermark"
            )

            generator = torch.Generator(device=_DEVICE)
            if seed is not None:
                generator = generator.manual_seed(seed)

            # Use appropriate dimensions based on device capability
            height, width = (512, 512) if _DEVICE == "cuda" else (384, 384)
            
            print(f"Starting Stable Diffusion inference...")
            print(f"Full prompt: {full_prompt}")
            
            img: Image.Image = _pipe(
                prompt=full_prompt,
                negative_prompt=negative,
                height=height, width=width,
                guidance_scale=7.5,
                num_inference_steps=25 if _DEVICE == "cuda" else 15,
                generator=generator
            ).images[0]
            
            print("Image generation completed successfully")

            # Resize to 512x512 if generated at different size
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.LANCZOS)

            img.save(cloth_path, "PNG")
            print(f"Saved generated cloth to {cloth_path}")

            # Auto-create mask using GrabCut
            auto_cloth_mask(cloth_path, mask_path)
            print(f"Created cloth mask at {mask_path}")
            
            # Verify files were created successfully
            if cloth_path.exists() and mask_path.exists():
                print(f"✅ Successfully generated cloth and mask")
                return cloth_path, mask_path
            else:
                raise FileNotFoundError("Generated files not found")
            
        except Exception as e:
            print(f"Error in Stable Diffusion generation: {e}")
            print("Falling back to simple generation...")

    # Fallback to simple generation
    return generate_cloth_from_prompt_v2(prompt, out_dir)

def generate_cloth_from_prompt_v2(prompt, output_dir):
    """Improved fallback cloth generator with better positioning."""
    # Create a 512x512 image
    img = Image.new('RGB', (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Determine colors based on prompt
    colors = {
        'red': (220, 50, 50),
        'blue': (50, 50, 220),
        'green': (50, 220, 50),
        'black': (50, 50, 50),
        'white': (240, 240, 240),
        'yellow': (240, 240, 50),
        'purple': (200, 50, 200),
        'pink': (255, 100, 150),
        'orange': (255, 150, 50),
        'gray': (150, 150, 150)
    }
    
    # Default color
    color = (100, 150, 200)
    
    # Check prompt for color keywords
    prompt_lower = prompt.lower()
    for color_name, color_val in colors.items():
        if color_name in prompt_lower:
            color = color_val
            break
    
    # Improved shirt positioning - centered and properly sized
    center_x, center_y = 256, 200  # Moved up for better torso positioning
    
    if 'shirt' in prompt_lower or 't-shirt' in prompt_lower:
        # Draw a more realistic t-shirt shape
        # Body of the shirt
        shirt_width = 180
        shirt_height = 200
        left = center_x - shirt_width // 2
        right = center_x + shirt_width // 2
        top = center_y - 40
        bottom = center_y + shirt_height - 40
        
        # Main body
        draw.rectangle([left, top, right, bottom], fill=color, outline=(0, 0, 0), width=2)
        
        # Sleeves
        sleeve_width = 60
        sleeve_height = 80
        # Left sleeve
        draw.rectangle([left - sleeve_width + 10, top, left + 20, top + sleeve_height], 
                      fill=color, outline=(0, 0, 0), width=2)
        # Right sleeve
        draw.rectangle([right - 20, top, right + sleeve_width - 10, top + sleeve_height], 
                      fill=color, outline=(0, 0, 0), width=2)
        
        # Neckline
        neck_width = 40
        neck_depth = 20
        draw.ellipse([center_x - neck_width//2, top - 5, center_x + neck_width//2, top + neck_depth], 
                    fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    
    elif 'dress' in prompt_lower:
        # Draw a dress shape
        dress_top_width = 120
        dress_bottom_width = 200
        dress_height = 280
        
        # Create dress silhouette points
        points = [
            (center_x - dress_top_width//2, center_y - 60),  # top left
            (center_x + dress_top_width//2, center_y - 60),  # top right
            (center_x + dress_bottom_width//2, center_y + dress_height - 60),  # bottom right
            (center_x - dress_bottom_width//2, center_y + dress_height - 60)   # bottom left
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
        
        # Add neckline
        neck_width = 30
        draw.ellipse([center_x - neck_width//2, center_y - 65, center_x + neck_width//2, center_y - 45], 
                    fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    
    else:
        # Default to shirt
        draw.rectangle([center_x - 90, center_y - 40, center_x + 90, center_y + 160], 
                      fill=color, outline=(0, 0, 0), width=2)
    
    # Save the generated cloth
    cloth_path = os.path.join(output_dir, "generated_cloth.png")
    img.save(cloth_path)
    
    # Create a corresponding mask
    mask = Image.new('L', (512, 512), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # Draw the same shape in white on the mask
    if 'shirt' in prompt_lower or 't-shirt' in prompt_lower:
        # Replicate the shirt shape for mask
        shirt_width = 180
        shirt_height = 200
        left = center_x - shirt_width // 2
        right = center_x + shirt_width // 2
        top = center_y - 40
        bottom = center_y + shirt_height - 40
        
        mask_draw.rectangle([left, top, right, bottom], fill=255)
        # Sleeves
        sleeve_width = 60
        sleeve_height = 80
        mask_draw.rectangle([left - sleeve_width + 10, top, left + 20, top + sleeve_height], fill=255)
        mask_draw.rectangle([right - 20, top, right + sleeve_width - 10, top + sleeve_height], fill=255)
    else:
        # Default mask
        mask_draw.rectangle([center_x - 90, center_y - 40, center_x + 90, center_y + 160], fill=255)
    
    mask_path = os.path.join(output_dir, "generated_cloth_mask.png")
    mask.save(mask_path)
    
    return cloth_path, mask_path

def generate_cloth_from_prompt(prompt, output_dir):
    """Generate clothing image from text prompt with robust error handling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if _pipe is None:
            print("Warning: Stable Diffusion not available, using fallback generator")
            return generate_cloth_from_prompt_v2(prompt, output_dir)
        
        # Enhanced prompt for better clothing generation
        enhanced_prompt = f"high quality fashion clothing, {prompt}, studio lighting, clean white background, product photography, detailed fabric texture"
        negative_prompt = "person, human, body, hands, face, wearing, model, mannequin, low quality, blurry, dark background, multiple items, text, watermark"
        
        # Generate with error handling
        try:
            print(f"Generating cloth with Stable Diffusion: {enhanced_prompt}")
            result = _pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25 if _DEVICE == "cuda" else 15,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            if result and hasattr(result, 'images') and len(result.images) > 0:
                cloth_img = result.images[0]
                
                # Save generated cloth
                cloth_path = output_dir / "gen_cloth.png"
                cloth_img.save(cloth_path)
                print(f"Generated cloth saved to: {cloth_path}")
                
                # Generate mask using auto_cloth_mask
                mask_path = output_dir / "gen_cloth_mask.png"
                auto_cloth_mask(cloth_path, mask_path)
                print(f"Generated mask saved to: {mask_path}")
                
                return str(cloth_path), str(mask_path)
            else:
                raise ValueError("No images generated from Stable Diffusion")
                
        except Exception as diffusion_error:
            print(f"Stable Diffusion inference failed: {diffusion_error}")
            print("Falling back to simple cloth generator")
            return generate_cloth_from_prompt_v2(prompt, output_dir)
            
    except Exception as e:
        print(f"Error in cloth generation: {e}")
        return generate_cloth_from_prompt_v2(prompt, output_dir)
