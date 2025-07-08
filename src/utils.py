import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_directories():
    """Create necessary directories"""
    dirs = ['data/training', 'data/test', 'data/samples',
            'models/saved_models', 'models/checkpoints',
            'results/training_samples', 'results/deblurred_images']

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created successfully!")


def download_sample_images():
    """Download some sample images for testing"""
    import urllib.request

    sample_urls = [
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
        "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800"
    ]

    for i, url in enumerate(sample_urls):
        try:
            urllib.request.urlretrieve(url, f"data/samples/sample_{i + 1}.jpg")
            print(f"Downloaded sample_{i + 1}.jpg")
        except Exception as e:
            print(f"Failed to download image {i + 1}: {e}")


def verify_installation():
    """Verify all packages are installed correctly"""
    packages = ['tensorflow', 'numpy', 'matplotlib', 'PIL', 'cv2', 'sklearn']

    for package in packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✓ {package} - version: {PIL._version_}")
            elif package == 'cv2':
                import cv2
                print(f"✓ {package} - version: {cv2._version_}")
            else:
                module = __import__(package)
                print(f"✓ {package} - version: {module._version_}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
        except AttributeError:
            print(f"✓ {package} - installed (version info not available)")