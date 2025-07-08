import sys
import os

sys.path.append('')
from utils import verify_installation, create_directories, download_sample_images

def main():
    print("=== Image Deblurring GAN Setup ===")

    # Verify installation
    print("\n1. Verifying installation...")
    verify_installation()

    # Create directories
    print("\n2. Creating project directories...")
    create_directories()

    # Download sample images
    print("\n3. Downloading sample images...")
    download_sample_images()

    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Add your training images to 'data/training/' folder")
    print("2. Run 'python src/training.py' to start training")
    print("3. Add test images to 'data/test/' folder")
    print("4. Run 'python src/test.py' to test the model")


if __name__ == "__main__":
        main()