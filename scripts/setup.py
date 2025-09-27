#!/usr/bin/env python3
"""
CreativeIQ Setup Script
Initializes the environment and downloads required models
"""

import os
import sys
import subprocess
import requests
from pathlib import Path


def print_step(step, message):
    """Print formatted step message"""
    print(f"\n{'='*50}")
    print(f"Step {step}: {message}")
    print(f"{'='*50}")


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True, capture_output=True, text=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def create_directories():
    """Create necessary directories"""
    directories = [
        "uploads",
        "models",
        "data/raw",
        "data/processed",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def setup_environment():
    """Set up environment file"""
    env_example = Path(".env.example")
    env_file = Path(".env")

    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        env_file.write_text(env_example.read_text())
        print("✓ Created .env file from template")
        print("⚠️  Please edit .env file with your configuration")
    else:
        print("✓ Environment file already exists")


def check_gpu_availability():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False


def download_test_images():
    """Download sample test images"""
    test_images_dir = Path("data/test_images")
    test_images_dir.mkdir(parents=True, exist_ok=True)

    # Sample image URLs (you can replace with actual test images)
    sample_images = [
        {
            "name": "sample_design.jpg",
            "url": "https://via.placeholder.com/800x600/4ECDC4/FFFFFF?text=Sample+Design"
        },
        {
            "name": "sample_poster.jpg",
            "url": "https://via.placeholder.com/600x800/FF6B6B/FFFFFF?text=Sample+Poster"
        }
    ]

    for image in sample_images:
        image_path = test_images_dir / image["name"]
        if not image_path.exists():
            try:
                response = requests.get(image["url"], timeout=10)
                response.raise_for_status()
                image_path.write_bytes(response.content)
                print(f"✓ Downloaded test image: {image['name']}")
            except requests.RequestException as e:
                print(f"⚠️  Could not download {image['name']}: {e}")


def run_basic_tests():
    """Run basic system tests"""
    try:
        # Test imports
        import fastapi
        import torch
        import cv2
        import PIL
        import transformers
        print("✓ All required packages can be imported")

        # Test FastAPI startup
        from app.main import app
        print("✓ FastAPI application loads successfully")

    except ImportError as e:
        print(f"⚠️  Import error: {e}")
    except Exception as e:
        print(f"⚠️  Application error: {e}")


def print_next_steps():
    """Print next steps for the user"""
    print(f"\n{'='*50}")
    print("Setup Complete! Next Steps:")
    print(f"{'='*50}")
    print("1. Edit .env file with your configuration:")
    print("   - Add HuggingFace token if needed")
    print("   - Configure database connections")
    print("   - Set DEVICE=cuda if you have GPU")
    print()
    print("2. Start the application:")
    print("   Option A (Docker): docker-compose up -d")
    print("   Option B (Local): uvicorn app.main:app --reload")
    print()
    print("3. Access the application:")
    print("   - Web Interface: http://localhost (Docker) or http://localhost:8000 (Local)")
    print("   - API Docs: http://localhost:8000/docs")
    print()
    print("4. Test with sample images in data/test_images/")
    print()
    print("For detailed documentation, see README.md")


def main():
    """Main setup function"""
    print("CreativeIQ Setup Script")
    print("Setting up your AI-powered design intelligence platform...")

    try:
        print_step(1, "Checking Python Version")
        check_python_version()

        print_step(2, "Installing Dependencies")
        install_dependencies()

        print_step(3, "Creating Directories")
        create_directories()

        print_step(4, "Setting Up Environment")
        setup_environment()

        print_step(5, "Checking GPU Availability")
        check_gpu_availability()

        print_step(6, "Downloading Test Images")
        download_test_images()

        print_step(7, "Running Basic Tests")
        run_basic_tests()

        print_next_steps()

    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()