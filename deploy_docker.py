#!/usr/bin/env python3
"""
Docker deployment script for Virtual Fashion Try-On
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ Docker found: {result.stdout.strip()}")
        
        # Check if Docker daemon is running
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        print("✅ Docker daemon is running")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found or not running")
        print("Please install Docker from: https://docs.docker.com/get-docker/")
        return False

def build_image(image_name="virtual-fashion-tryon", tag="latest"):
    """Build Docker image"""
    print(f"🔨 Building Docker image: {image_name}:{tag}")
    
    try:
        # Build the image
        cmd = ["docker", "build", "-t", f"{image_name}:{tag}", "."]
        process = subprocess.run(cmd, check=True)
        
        print(f"✅ Image built successfully: {image_name}:{tag}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

def run_container(image_name="virtual-fashion-tryon", tag="latest", port=8000):
    """Run Docker container"""
    container_name = f"{image_name}-container"
    
    print(f"🚀 Starting container: {container_name}")
    
    try:
        # Stop existing container if running
        subprocess.run([
            "docker", "stop", container_name
        ], capture_output=True)
        
        subprocess.run([
            "docker", "rm", container_name
        ], capture_output=True)
        
        # Run new container
        cmd = [
            "docker", "run",
            "--name", container_name,
            "-p", f"{port}:8000",
            "-d",  # Detached mode
            f"{image_name}:{tag}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        
        print(f"✅ Container started: {container_id[:12]}")
        print(f"🌐 Application available at: http://localhost:{port}")
        
        return container_id
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start container: {e}")
        return None

def show_logs(container_name="virtual-fashion-tryon-container"):
    """Show container logs"""
    print(f"📋 Container logs for {container_name}:")
    try:
        subprocess.run(["docker", "logs", "-f", container_name], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to show logs")
    except KeyboardInterrupt:
        print("\n📋 Log viewing stopped")

def check_health(port=8000, max_retries=30):
    """Check if the application is healthy"""
    print(f"🔍 Checking application health on port {port}...")
    
    import urllib.request
    import urllib.error
    
    for i in range(max_retries):
        try:
            response = urllib.request.urlopen(f"http://localhost:{port}", timeout=5)
            if response.status == 200:
                print("✅ Application is healthy and responding")
                return True
        except urllib.error.URLError:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Attempt {i+1}/{max_retries} - waiting...")
            time.sleep(2)
    
    print("❌ Application health check failed")
    return False

def cleanup():
    """Clean up Docker resources"""
    container_name = "virtual-fashion-tryon-container"
    image_name = "virtual-fashion-tryon"
    
    print("🧹 Cleaning up Docker resources...")
    
    # Stop container
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)
    
    # Remove image (optional)
    choice = input(f"Remove Docker image '{image_name}'? (y/N): ").lower()
    if choice == 'y':
        subprocess.run(["docker", "rmi", image_name], capture_output=True)
        print(f"✅ Removed image: {image_name}")
    
    print("✅ Cleanup completed")

def main():
    """Main deployment function"""
    print("🐳 Docker Deployment for Virtual Fashion Try-On")
    print("=" * 50)
    
    # Check Docker
    if not check_docker():
        return
    
    # Check required files
    if not os.path.exists("Dockerfile"):
        print("❌ Dockerfile not found!")
        return
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    print("\nChoose an option:")
    print("1. Build and run (full deployment)")
    print("2. Build only")
    print("3. Run existing image")
    print("4. Show logs")
    print("5. Cleanup")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        # Full deployment
        if build_image():
            container_id = run_container()
            if container_id:
                time.sleep(5)  # Wait for startup
                if check_health():
                    print("\n🎉 Deployment successful!")
                    print("🌐 Visit: http://localhost:8000")
                    
                    # Ask if user wants to see logs
                    if input("\nShow logs? (y/N): ").lower() == 'y':
                        show_logs()
                else:
                    print("❌ Application is not responding")
    
    elif choice == "2":
        # Build only
        build_image()
    
    elif choice == "3":
        # Run existing
        container_id = run_container()
        if container_id and check_health():
            print("🎉 Container running successfully!")
    
    elif choice == "4":
        # Show logs
        show_logs()
    
    elif choice == "5":
        # Cleanup
        cleanup()
    
    elif choice == "6":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()