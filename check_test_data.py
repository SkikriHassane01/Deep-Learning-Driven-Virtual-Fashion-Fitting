#!/usr/bin/env python3
"""
Script to check and fix the test data directory structure
"""
import os
from pathlib import Path

def check_test_data_structure(testZdata_dir):
    """Check what directories exist in the test data"""
    test_data_dir = Path(test_data_dir)
    
    print(f"🔍 Checking test data structure in: {test_data_dir}")
    print("=" * 50)
    
    if not test_data_dir.exists():
        print(f"❌ Test data directory doesn't exist: {test_data_dir}")
        return
    
    # List all subdirectories
    subdirs = [d for d in test_data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories:")
    
    for subdir in sorted(subdirs):
        files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + list(subdir.glob("*.json"))
        print(f"  📁 {subdir.name}: {len(files)} files")
        
        # Show first few files as example
        if files:
            print(f"     Examples: {', '.join([f.name for f in files[:3]])}")
    
    # Check for test pairs file
    test_pairs_file = test_data_dir / "test_pairs.txt"
    if test_pairs_file.exists():
        with open(test_pairs_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"\n📄 Test pairs file: {len(lines)} pairs")
        if lines:
            print(f"     Example: {lines[0]}")
    else:
        print(f"\n❌ No test_pairs.txt found in {test_data_dir}")
        
        # Look for it in parent directory
        parent_pairs = test_data_dir.parent / "test_pairs.txt"
        if parent_pairs.exists():
            print(f"✅ Found test_pairs.txt in parent directory: {parent_pairs}")

def create_sample_test_data(base_dir):
    """Create a small sample test dataset for evaluation"""
    base_dir = Path(base_dir)
    
    # Look for existing data in the project
    possible_data_dirs = [
        base_dir / "data",
        base_dir / "media" / "uploads",
    ]
    
    person_images = []
    cloth_images = []
    
    for data_dir in possible_data_dirs:
        if data_dir.exists():
            # Look for person images
            for subdir in ["image", "user", "person"]:
                img_dir = data_dir / subdir
                if img_dir.exists():
                    imgs = list(img_dir.glob("*.jpg"))[:10]  # Take first 10
                    person_images.extend(imgs)
            
            # Look for cloth images
            for subdir in ["cloth", "clothing"]:
                cloth_dir = data_dir / subdir
                if cloth_dir.exists():
                    imgs = list(cloth_dir.glob("*.jpg"))[:10]  # Take first 10
                    cloth_images.extend(imgs)
    
    if not person_images or not cloth_images:
        print("❌ No images found to create sample test data")
        return
    
    # Create test structure
    test_dir = base_dir / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Create directories
    (test_dir / "image").mkdir(exist_ok=True)
    (test_dir / "cloth").mkdir(exist_ok=True)
    (test_dir / "cloth-mask").mkdir(exist_ok=True)
    
    # Copy sample images
    import shutil
    
    print(f"📦 Creating sample test data in {test_dir}")
    
    copied_persons = []
    copied_clothes = []
    
    # Copy person images
    for i, img_path in enumerate(person_images[:5]):  # Take 5 samples
        dest_path = test_dir / "image" / f"person_{i:03d}.jpg"
        shutil.copy2(img_path, dest_path)
        copied_persons.append(dest_path.name)
        print(f"  📷 Copied person image: {dest_path.name}")
    
    # Copy cloth images
    for i, img_path in enumerate(cloth_images[:5]):  # Take 5 samples
        dest_path = test_dir / "cloth" / f"cloth_{i:03d}.jpg"
        shutil.copy2(img_path, dest_path)
        copied_clothes.append(dest_path.name)
        print(f"  👕 Copied cloth image: {dest_path.name}")
    
    # Create test pairs
    pairs = []
    for i in range(min(len(copied_persons), len(copied_clothes))):
        pairs.append(f"{copied_persons[i]} {copied_clothes[i]}")
    
    pairs_file = test_dir / "test_pairs.txt"
    with open(pairs_file, 'w') as f:
        f.write('\n'.join(pairs))
    
    print(f"📝 Created test pairs file with {len(pairs)} pairs")
    print(f"✅ Sample test data created at: {test_dir}")
    return test_dir

if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent
    
    if len(sys.argv) > 1:
        test_data_dir = sys.argv[1]
    else:
        # Try common locations
        possible_dirs = [
            project_root / "data" / "test",
            project_root / "test_data",
            project_root / "data",
        ]
        
        test_data_dir = None
        for d in possible_dirs:
            if d.exists():
                test_data_dir = d
                break
        
        if not test_data_dir:
            print("No test data directory found. Creating sample test data...")
            test_data_dir = create_sample_test_data(project_root)
    
    if test_data_dir:
        check_test_data_structure(test_data_dir)