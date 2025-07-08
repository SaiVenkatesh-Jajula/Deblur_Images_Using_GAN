import os
import glob
import sys

sys.path.append('src')

from improved_deblur_gan import ImprovedDeblurGAN
from utils import create_directories


def main():
    print("=== Improved Image Deblurring GAN Training ===")

    # Create directories
    create_directories()

    # Initialize the improved GAN
    print("\nInitializing Improved DeblurGAN...")
    deblur_gan = ImprovedDeblurGAN()

    # Get training images
    train_path = "/Users/hvtsa/PycharmProjects/ImageDeblur/data/training"
    if not os.path.exists(train_path) or len(os.listdir(train_path)) == 0:
        print(f"\n❌ No images found in {train_path}")
        print("Please add some images to the data/train folder")
        print("\nYou can download sample images from:")
        print("- https://unsplash.com")
        print("- https://www.pexels.com")
        print("- Your own photos")
        return

    # Get all image files - FIXED: Added missing asterisk
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(train_path, '*' + ext)))
        image_paths.extend(glob.glob(os.path.join(train_path, '*' + ext.upper())))

    if len(image_paths) == 0:
        print("❌ No valid images found!")
        return

    print(f"✅ Found {len(image_paths)} training images")

    # Training parameters - IMPROVED
    training_params = {
        'epochs': 25,  # Reduced but more effective
        'batch_size': 4,  # Good balance
        'save_interval': 25,  # FIXED: Changed from 25 to 5
        'validation_split': 0.2  # 20% for validation
    }

    print(f"\n📋 Training Parameters:")
    for param, value in training_params.items():
        print(f"   {param}: {value}")

    print(f"\n🚀 Starting training...")
    print("This will take approximately:")
    print(f"   - With GPU: 30-60 minutes")
    print(f"   - With CPU: 2-4 hours")
    print("\nProgress will be shown below:")
    print("-" * 50)

    # Start training
    try:
        history = deblur_gan.train(
            image_paths=image_paths,
            **training_params
        )

        print("\n" + "=" * 50)
        print("🎉 Training completed successfully!")
        print("=" * 50)

        # Show final results
        final_g_loss = history['g_loss'][-1] if history['g_loss'] else 0
        final_d_loss = history['d_loss'][-1] if history['d_loss'] else 0

        print(f"\n📊 Final Results:")
        print(f"   Generator Loss: {final_g_loss:.4f}")
        print(f"   Discriminator Loss: {final_d_loss:.4f}")

        print(f"\n💾 Models saved in: models/saved_models/")
        print(f"📸 Sample images in: results/training_samples/")
        print(f"📈 Training history plot saved as: training_curves.png")

        print(f"\n🔥 Next steps:")
        print(f"1. Check the sample images to see improvement")
        print(f"2. Test on new images using the test script")
        print(f"3. If results are good, you can continue training with more epochs")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error trace
        print("\n🔧 Troubleshooting:")
        print("1. Reduce batch_size to 2 if you get memory errors")
        print("2. Reduce epochs to 10 for quick testing")
        print("3. Make sure you have at least 10 training images")
        print("4. Check if your images are valid (not corrupted)")


def quick_test():
    """Quick test with minimal parameters"""
    print("=== Quick Test Mode ===")

    deblur_gan = ImprovedDeblurGAN()

    train_path = "/Users/hvtsa/PycharmProjects/ImageDeblur/data/training"  # FIXED: Updated path
    image_paths = glob.glob(os.path.join(train_path, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(train_path, "*.png")))
    image_paths.extend(glob.glob(os.path.join(train_path, "*.jpeg")))

    if len(image_paths) < 5:
        print("❌ Need at least 5 images for quick test")
        return

    print(f"🚀 Quick test with {len(image_paths)} images")

    # Minimal parameters for testing
    history = deblur_gan.train(
        image_paths=image_paths,
        epochs=3,  # Very few epochs
        batch_size=2,  # Small batch
        save_interval=3,  # Save at the end
        validation_split=0.0  # No validation for quick test
    )

    print("✅ Quick test completed!")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()