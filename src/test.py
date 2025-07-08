import os
import glob
from improved_deblur_gan import ImprovedDeblurGAN


def main():
    # Initialize the GAN
    deblur_gan = ImprovedDeblurGAN()

    # Load trained model
    model_path = "C:/Users/hvtsa/PycharmProjects/ImageDeblur/generator_epoch_10.h5"
    if not os.path.exists(model_path):
        print("No trained model found!")
        print("Available models:")
        for model in glob.glob("models/saved_models/*.h5"):
            print(f"  - {model}")
        return

    deblur_gan.load_models(model_path)

    # Test on sample images
    test_images = glob.glob("data/test/.jpg") + glob.glob("data/test/.png")

    if not test_images:
        print("No test images found in data/test/")
        return

    for img_path in test_images:
        output_path = f"results/deblurred_images/deblurred_{os.path.basename(img_path)}"
        deblur_gan.deblur_image(img_path, output_path)


if __name__ == "__main__":
        main()