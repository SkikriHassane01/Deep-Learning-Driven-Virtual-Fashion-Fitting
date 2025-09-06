#!/usr/bin/env python3
"""
Garment Generation Pipeline Entry Point
Simple script that runs the complete garment generation workflow using existing modules
"""

import os
import sys

# Add src to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from models.garment_generator import SimpleGarmentGenerator as GarmentGenerator


def main():
    """Main execution pipeline for garment generation"""
    
    print("=" * 60)
    print("👗 GARMENT GENERATION PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("outputs/garment_generation", exist_ok=True)
    
    # Initialize garment generator
    print("\n🔄 Initializing Garment Generator...")
    generator = GarmentGenerator()
    
    # Test prompts like in the notebook
    test_prompts = [
        "elegant red evening dress",
        "casual blue jeans with white t-shirt"
    ]
    
    print(f"\n🎨 Generating garments for {len(test_prompts)} prompts...")
    
    results = []
    
    # Generate garments for each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Generation {i}/{len(test_prompts)} ---")
        print(f"Prompt: '{prompt}'")
        
        try:
            # Generate garment
            generated_image = generator.generate_garment(prompt)
            
            # Save result
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"outputs/garment_generation/garment_{i}_{safe_prompt.replace(' ', '_')}.png"
            generated_image.save(filename)
            
            results.append({
                'prompt': prompt,
                'image': generated_image,
                'filename': filename
            })
            
            print(f"✅ Successfully generated and saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating '{prompt}': {e}")
    
    print(f"\n{'=' * 60}")
    print("🎉 GARMENT GENERATION COMPLETE!")
    print(f"✅ Generated {len(results)}/{len(test_prompts)} garments successfully")
    print(f"📁 Results saved to: outputs/garment_generation/")
    
    for result in results:
        print(f"  - {result['filename']}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()