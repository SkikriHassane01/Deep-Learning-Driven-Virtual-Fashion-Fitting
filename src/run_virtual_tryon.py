#!/usr/bin/env python3
"""
Virtual Try-On Pipeline Entry Point
Simple script that runs the complete virtual try-on workflow using existing modules
"""

import os
import sys
import requests
from PIL import Image

# Add src to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from models.virtual_tryon import VirtualTryOnPipeline


def download_test_image():
    """Download test image from the web"""
    test_url = "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    
    try:
        print("📥 Downloading test image...")
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        
        # Save test image
        image_path = "outputs/virtual_tryon/test_image.jpg"
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        # Load and resize image
        img = Image.open(image_path)
        img = img.resize((512, 512))
        img.save(image_path)
        
        print(f"✅ Test image downloaded and resized: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"❌ Failed to download test image: {e}")
        return None


def main():
    """Main execution pipeline for virtual try-on"""
    
    print("=" * 60)
    print("🎭 VIRTUAL TRY-ON PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("outputs/virtual_tryon", exist_ok=True)
    
    # Initialize virtual try-on system
    print("\n🔄 Initializing Virtual Try-On System...")
    tryon_system = VirtualTryOnPipeline()
    
    # Download test image
    image_path = download_test_image()
    if not image_path:
        print("❌ Cannot proceed without test image")
        return
    
    # Load test image
    test_image = Image.open(image_path).convert('RGB')
    
    # Test prompts like in the notebook
    test_prompts = [
        "Men's Nick Standard Fit T-Shirt",
        "casual blue denim jacket with silver buttons"
    ]
    
    print(f"\n🎨 Processing virtual try-on for {len(test_prompts)} prompts...")
    
    results = []
    
    # Process each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Try-On {i}/{len(test_prompts)} ---")
        print(f"Prompt: '{prompt}'")
        
        try:
            # Process virtual try-on
            result = tryon_system.process_single_prompt(test_image, prompt)
            
            # Save result
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"outputs/virtual_tryon/tryon_{i}_{safe_prompt.replace(' ', '_')}.png"
            result['result'].save(filename)
            
            results.append({
                'prompt': prompt,
                'result': result,
                'filename': filename
            })
            
            print(f"✅ Successfully processed and saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing '{prompt}': {e}")
    
    print(f"\n{'=' * 60}")
    print("🎉 VIRTUAL TRY-ON COMPLETE!")
    print(f"✅ Processed {len(results)}/{len(test_prompts)} try-ons successfully")
    print(f"📁 Results saved to: outputs/virtual_tryon/")
    
    for result in results:
        print(f"  - {result['filename']}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()