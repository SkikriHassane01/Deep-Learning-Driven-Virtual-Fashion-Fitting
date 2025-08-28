# Virtual Try-On Test Script
# filepath: d:\03. Projects\technocolabs_projects\VirtualFashion\test_tryon.py

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def create_test_images():
    """Create simple test images for debugging."""
    test_dir = project_root / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple person image (512x512)
    person_img = Image.new('RGB', (512, 512), (200, 180, 160))  # Skin tone
    draw = ImageDraw.Draw(person_img)
    
    # Draw a simple human figure
    # Head
    draw.ellipse([230, 50, 280, 100], fill=(220, 200, 180))
    # Body
    draw.rectangle([220, 100, 290, 300], fill=(100, 100, 150))  # Blue shirt
    # Arms
    draw.rectangle([190, 120, 220, 200], fill=(220, 200, 180))  # Left arm
    draw.rectangle([290, 120, 320, 200], fill=(220, 200, 180))  # Right arm
    # Legs
    draw.rectangle([230, 300, 260, 450], fill=(50, 50, 100))    # Left leg
    draw.rectangle([260, 300, 290, 450], fill=(50, 50, 100))    # Right leg
    
    person_path = test_dir / "person.jpg"
    person_img.save(person_path)
    
    # Create a simple cloth image
    cloth_img = Image.new('RGB', (512, 512), (255, 255, 255))  # White background
    draw = ImageDraw.Draw(cloth_img)
    
    # Draw a red shirt
    draw.rectangle([150, 100, 360, 300], fill=(200, 50, 50), outline=(0, 0, 0), width=2)
    # Sleeves
    draw.rectangle([100, 120, 150, 200], fill=(200, 50, 50), outline=(0, 0, 0), width=2)
    draw.rectangle([360, 120, 410, 200], fill=(200, 50, 50), outline=(0, 0, 0), width=2)
    
    cloth_path = test_dir / "cloth.jpg"
    cloth_img.save(cloth_path)
    
    print(f"✅ Created test images:")
    print(f"   Person: {person_path}")
    print(f"   Cloth: {cloth_path}")
    
    return str(person_path), str(cloth_path)

def test_pose_extraction(person_path):
    """Test pose extraction."""
    try:
        from tryon.services.pose_seg import process_user_image
        
        output_dir = Path("test_data/pose_output")
        pose_json, body_mask = process_user_image(person_path, output_dir)
        
        print(f"✅ Pose extraction successful:")
        print(f"   Pose JSON: {pose_json}")
        print(f"   Body mask: {body_mask}")
        
        return pose_json, body_mask
    except Exception as e:
        print(f"❌ Pose extraction failed: {e}")
        return None, None

def test_cloth_mask(cloth_path):
    """Test cloth mask generation."""
    try:
        from tryon.services.auto_mask import auto_cloth_mask
        
        mask_path = Path("test_data/cloth_mask.png")
        result = auto_cloth_mask(cloth_path, mask_path)
        
        print(f"✅ Cloth mask generation successful:")
        print(f"   Mask: {result}")
        
        return result
    except Exception as e:
        print(f"❌ Cloth mask generation failed: {e}")
        return None

def test_fitting(person_path, cloth_path, pose_json, cloth_mask):
    """Test the actual fitting process."""
    try:
        # Mock Django settings
        class MockSettings:
            MEDIA_ROOT = str(Path("test_data/results"))
        
        import sys
        sys.modules['django.conf'] = type('MockModule', (), {'settings': MockSettings()})()
        
        from tryon.services.fitter import baseline_fit
        
        # Ensure output directory exists
        Path(MockSettings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
        
        result_path = baseline_fit(person_path, cloth_path, pose_json, cloth_mask)
        
        print(f"✅ Virtual try-on successful:")
        print(f"   Result: {result_path}")
        
        return result_path
    except Exception as e:
        print(f"❌ Virtual try-on failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_text_to_cloth(prompt="red t-shirt"):
    """Test text-to-cloth generation."""
    try:
        from tryon.services.text2cloth import generate_cloth_from_prompt
        
        output_dir = Path("test_data/text2cloth")
        cloth_path, mask_path = generate_cloth_from_prompt(prompt, output_dir)
        
        print(f"✅ Text-to-cloth generation successful:")
        print(f"   Cloth: {cloth_path}")
        print(f"   Mask: {mask_path}")
        
        return cloth_path, mask_path
    except Exception as e:
        print(f"❌ Text-to-cloth generation failed: {e}")
        return None, None

def main():
    print("🧪 Testing Virtual Try-On System")
    print("=" * 50)
    
    # Create test images
    person_path, cloth_path = create_test_images()
    
    # Test pose extraction
    print("\n📍 Testing pose extraction...")
    pose_json, body_mask = test_pose_extraction(person_path)
    
    # Test cloth mask generation
    print("\n👔 Testing cloth mask generation...")
    cloth_mask = test_cloth_mask(cloth_path)
    
    # Test text-to-cloth generation
    print("\n📝 Testing text-to-cloth generation...")
    text_cloth, text_mask = test_text_to_cloth("blue dress")
    
    # Test virtual try-on with original cloth
    print("\n🎯 Testing virtual try-on (original cloth)...")
    result1 = test_fitting(person_path, cloth_path, pose_json, cloth_mask)
    
    # Test virtual try-on with generated cloth
    if text_cloth:
        print("\n🎯 Testing virtual try-on (generated cloth)...")
        result2 = test_fitting(person_path, text_cloth, pose_json, text_mask)
    
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    print("=" * 50)
    
    if result1:
        print("✅ Virtual try-on with original cloth: PASSED")
    else:
        print("❌ Virtual try-on with original cloth: FAILED")
    
    if text_cloth and result2:
        print("✅ Virtual try-on with generated cloth: PASSED")
    elif text_cloth:
        print("❌ Virtual try-on with generated cloth: FAILED")
    else:
        print("⚠️ Text-to-cloth generation failed, skipping second test")
    
    print(f"\n📁 Check results in: {Path('test_data').absolute()}")

if __name__ == "__main__":
    main()