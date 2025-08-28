#!/usr/bin/env python3
"""
Test script to verify the virtual try-on fixes are working correctly.
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vfit.settings')
django.setup()

from tryon.services.pose_seg import process_user_image, extract_pose_keypoints
from tryon.services.fitter import baseline_fit
from tryon.services.auto_mask import auto_cloth_mask
import cv2
import numpy as np
from PIL import Image
import json

def test_pose_estimation():
    """Test pose estimation fixes."""
    print("=== Testing Pose Estimation ===")
    
    # Create a simple test image
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    test_img[100:400, 200:300] = [128, 128, 128]  # Simple person silhouette
    
    # Save test image
    test_path = "test_person.jpg"
    cv2.imwrite(test_path, test_img)
    
    try:
        # Test pose keypoint extraction
        keypoints = extract_pose_keypoints(test_img)
        print(f"✅ Pose estimation working - extracted {len(keypoints)} keypoints")
        
        # Test full user image processing
        pose_json, body_mask = process_user_image(test_path, "test_output")
        print(f"✅ User image processing complete")
        print(f"   - Pose JSON: {pose_json}")
        print(f"   - Body mask: {body_mask}")
        
        # Verify files were created
        if os.path.exists(pose_json):
            with open(pose_json, 'r') as f:
                pose_data = json.load(f)
                print(f"   - Pose data contains {len(pose_data.get('keypoints', []))} keypoints")
        
        if os.path.exists(body_mask):
            print(f"   - Body mask file created successfully")
        
    except Exception as e:
        print(f"❌ Pose estimation failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)

def test_cloth_masking():
    """Test cloth mask generation."""
    print("\n=== Testing Cloth Masking ===")
    
    # Create a simple cloth image
    cloth_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
    cloth_path = "test_cloth.jpg"
    
    try:
        cv2.imwrite(cloth_path, cloth_img)
        
        # Test mask generation
        mask_path = auto_cloth_mask(cloth_path, "test_cloth_mask.png")
        print(f"✅ Cloth mask generation working")
        print(f"   - Mask saved to: {mask_path}")
        
        if os.path.exists(mask_path):
            print(f"   - Mask file created successfully")
            
    except Exception as e:
        print(f"❌ Cloth masking failed: {e}")
    finally:
        # Cleanup
        for f in [cloth_path, "test_cloth_mask.png"]:
            if os.path.exists(f):
                os.remove(f)

def test_try_on_pipeline():
    """Test the complete try-on pipeline."""
    print("\n=== Testing Try-On Pipeline ===")
    
    # Create test images
    person_img = np.zeros((512, 512, 3), dtype=np.uint8)
    person_img[100:400, 200:300] = [150, 150, 150]  # Person silhouette
    
    cloth_img = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
    
    person_path = "test_person_pipeline.jpg"
    cloth_path = "test_cloth_pipeline.jpg"
    
    try:
        cv2.imwrite(person_path, person_img)
        cv2.imwrite(cloth_path, cloth_img)
        
        # Test the complete pipeline
        result_path = baseline_fit(
            person_image_path=person_path,
            cloth_image_path=cloth_path,
            pose_json_path=None,
            cloth_mask_path=None
        )
        
        print(f"✅ Try-on pipeline working")
        print(f"   - Result saved to: {result_path}")
        
        if os.path.exists(result_path):
            # Verify the result is a valid image
            result_img = Image.open(result_path)
            print(f"   - Result image size: {result_img.size}")
            print(f"   - Result image mode: {result_img.mode}")
            
    except Exception as e:
        print(f"❌ Try-on pipeline failed: {e}")
    finally:
        # Cleanup
        for f in [person_path, cloth_path]:
            if os.path.exists(f):
                os.remove(f)

def test_file_organization():
    """Test file organization and saving."""
    print("\n=== Testing File Organization ===")
    
    from django.conf import settings
    
    # Check media directories
    media_root = Path(settings.MEDIA_ROOT)
    print(f"Media root: {media_root}")
    
    # Test directory creation
    test_dirs = [
        media_root / "outputs" / "test" / "generated",
        media_root / "outputs" / "test" / "pose", 
        media_root / "outputs" / "test" / "masks",
        media_root / "results"
    ]
    
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path.exists():
            print(f"✅ Directory created: {dir_path}")
        else:
            print(f"❌ Failed to create: {dir_path}")

def main():
    """Run all tests."""
    print("🔧 Testing Virtual Try-On Fixes")
    print("=" * 50)
    
    test_pose_estimation()
    test_cloth_masking()
    test_try_on_pipeline()
    test_file_organization()
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")
    print("\nKey fixes implemented:")
    print("• Fixed OpenPose keypoint extraction with better error handling")
    print("• Enhanced virtual try-on model with fallback methods")
    print("• Improved file organization and saving to correct directories")
    print("• Added intelligent warping and blending algorithms")
    print("• Better pose data processing and mask generation")

if __name__ == "__main__":
    main()
