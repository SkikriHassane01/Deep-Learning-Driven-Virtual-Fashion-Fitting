#!/usr/bin/env python3
"""
Deployment script for Hugging Face Spaces
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "app.py",
        "requirements.txt",
        "webapp/manage.py",
        "src/"
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def create_hf_structure():
    """Create Hugging Face deployment structure"""
    print("📁 Creating Hugging Face deployment structure...")
    
    # Create deployment directory
    deploy_dir = Path("hf_deploy")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Copy essential files
    files_to_copy = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("README_HF.md", "README.md"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, deploy_dir / dst)
            print(f"   ✅ Copied {src} → {dst}")
        else:
            print(f"   ⚠️  Missing {src}")
    
    # Copy webapp directory
    if os.path.exists("webapp"):
        shutil.copytree("webapp", deploy_dir / "webapp")
        print("   ✅ Copied webapp/")
    
    # Copy src directory
    if os.path.exists("src"):
        shutil.copytree("src", deploy_dir / "src")
        print("   ✅ Copied src/")
    
    # Copy model directory (if exists and not too large)
    if os.path.exists("models"):
        model_size = get_directory_size("models")
        if model_size < 1000 * 1024 * 1024:  # Less than 1GB
            shutil.copytree("models", deploy_dir / "models")
            print(f"   ✅ Copied models/ ({model_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"   ⚠️  Models directory too large ({model_size / 1024 / 1024:.1f} MB)")
            print("      Consider using Git LFS or external hosting")
    
    print(f"🎉 Deployment structure created in {deploy_dir}/")
    return deploy_dir

def get_directory_size(path):
    """Get total size of directory in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total

def install_hf_cli():
    """Install Hugging Face CLI if not present"""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        print("✅ Hugging Face CLI already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing Hugging Face CLI...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            print("✅ Hugging Face CLI installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Hugging Face CLI")
            return False

def create_git_repo(deploy_dir, space_name):
    """Initialize git repo and add files"""
    print(f"🔄 Setting up Git repository in {deploy_dir}...")
    
    os.chdir(deploy_dir)
    
    try:
        # Initialize git repo
        subprocess.run(["git", "init"], check=True)
        
        # Add Hugging Face remote
        remote_url = f"https://huggingface.co/spaces/{space_name}"
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        
        # Add files
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit
        subprocess.run([
            "git", "commit", "-m", 
            "Initial deployment of Virtual Fashion Try-On app"
        ], check=True)
        
        print("✅ Git repository set up successfully")
        print(f"🔗 Remote: {remote_url}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git setup failed: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Starting Hugging Face Spaces deployment...")
    
    # Check requirements
    if not check_requirements():
        print("❌ Deployment aborted - missing requirements")
        return
    
    # Get space name
    space_name = input("Enter your Hugging Face Space name (username/space-name): ").strip()
    if not space_name or "/" not in space_name:
        print("❌ Invalid space name. Use format: username/space-name")
        return
    
    # Install HF CLI
    if not install_hf_cli():
        print("❌ Cannot proceed without Hugging Face CLI")
        return
    
    # Create deployment structure
    deploy_dir = create_hf_structure()
    
    # Setup git repository
    original_dir = os.getcwd()
    try:
        if create_git_repo(deploy_dir, space_name):
            print("\n🎉 Deployment prepared successfully!")
            print("\nNext steps:")
            print(f"1. cd {deploy_dir}")
            print("2. huggingface-cli login")
            print("3. git push origin main")
            print("\nOr manually upload files to:")
            print(f"https://huggingface.co/spaces/{space_name}")
        else:
            print("❌ Git setup failed - check manually")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()