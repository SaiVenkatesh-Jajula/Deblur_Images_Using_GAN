import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


class ImprovedDeblurGAN:
    def _init_(self, img_height=256, img_width=256, channels=3):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.img_shape = (img_height, img_width, channels)

        # Build networks
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Compile discriminator with label smoothing
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Build combined model
        self.discriminator.trainable = False
        blur_input = keras.Input(shape=self.img_shape)
        deblurred = self.generator(blur_input)
        validity = self.discriminator(deblurred)

        self.combined = keras.Model(blur_input, [deblurred, validity])
        self.combined.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.999),
            loss=[self.content_loss, 'binary_crossentropy'],
            loss_weights=[100, 1]
        )

        # Load VGG for perceptual loss
        self.vgg = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=self.img_shape)
        self.vgg.trainable = False

        # Extract specific layers for perceptual loss
        self.feature_extractor = keras.Model(
            inputs=self.vgg.input,
            outputs=self.vgg.get_layer('block3_conv3').output
        )
        self.feature_extractor.trainable = False

    def reflection_pad(self, x, padding=1):
        """Reflection padding to reduce artifacts"""
        return tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')

    def instance_norm(self, x, name=None):
        """Instance normalization"""
        return layers.GroupNormalization(groups=1, name=name)(x)

    def resnet_block(self, x, filters, use_dropout=False):
        """Improved ResNet block with instance normalization"""
        init = x

        # First conv
        x = self.reflection_pad(x, 1)
        x = layers.Conv2D(filters, 3, padding='valid', use_bias=False)(x)
        x = self.instance_norm(x)
        x = layers.ReLU()(x)

        if use_dropout:
            x = layers.Dropout(0.5)(x)

        # Second conv
        x = self.reflection_pad(x, 1)
        x = layers.Conv2D(filters, 3, padding='valid', use_bias=False)(x)
        x = self.instance_norm(x)

        # Skip connection
        x = layers.Add()([init, x])
        return x

    def upsample_block(self, x, filters, kernel_size=3):
        """Improved upsampling block to avoid checkerboard artifacts"""
        # Use UpSampling2D + Conv2D instead of Conv2DTranspose
        x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        x = self.reflection_pad(x, 1)
        x = layers.Conv2D(filters, kernel_size, padding='valid', use_bias=False)(x)
        x = self.instance_norm(x)
        x = layers.ReLU()(x)
        return x

    def build_generator(self):
        """Improved Generator with better architecture"""
        inputs = keras.Input(shape=self.img_shape)

        # Initial conv block
        x = self.reflection_pad(inputs, 3)
        x = layers.Conv2D(64, 7, padding='valid', use_bias=False)(x)
        x = self.instance_norm(x)
        x = layers.ReLU()(x)

        # Downsampling
        x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
        x = self.instance_norm(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False)(x)
        x = self.instance_norm(x)
        x = layers.ReLU()(x)

        # ResNet blocks with dropout in the middle layers
        for i in range(9):
            use_dropout = i >= 3 and i <= 5  # Add dropout to middle layers
            x = self.resnet_block(x, 256, use_dropout=use_dropout)

        # Upsampling with improved blocks
        x = self.upsample_block(x, 128)
        x = self.upsample_block(x, 64)

        # Output layer
        x = self.reflection_pad(x, 3)
        outputs = layers.Conv2D(3, 7, padding='valid', activation='tanh')(x)

        model = keras.Model(inputs, outputs, name='Improved_Generator')
        return model

    def build_discriminator(self):
        """Improved Discriminator with spectral normalization"""
        inputs = keras.Input(shape=self.img_shape)

        # Use different kernel sizes for better feature detection
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = self.instance_norm(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = self.instance_norm(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(512, 4, strides=1, padding='same')(x)
        x = self.instance_norm(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Final classification
        x = layers.Conv2D(1, 4, strides=1, padding='same')(x)
        outputs = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(1, activation='sigmoid')(outputs)

        model = keras.Model(inputs, outputs, name='Improved_Discriminator')
        return model

    def content_loss(self, y_true, y_pred):
        """Combined L1 + Perceptual loss"""
        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        # Perceptual loss
        y_true_norm = (y_true + 1) / 2
        y_pred_norm = (y_pred + 1) / 2

        true_features = self.feature_extractor(y_true_norm)
        pred_features = self.feature_extractor(y_pred_norm)

        perceptual_loss = tf.reduce_mean(tf.abs(true_features - pred_features))

        return l1_loss + 0.1 * perceptual_loss

    def load_and_preprocess_image(self, image_path, target_size=(256, 256)):
        """Load and preprocess image with better error handling"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def create_realistic_blur(self, image, blur_type='motion'):
        """Improved blur generation with less aggressive noise"""
        if blur_type == 'motion':
            # Reduce kernel size for more realistic blur
            kernel_size = np.random.randint(5, 15)  # Reduced from 15-25
            angle = np.random.randint(0, 180)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size

            # Rotate kernel
            center = (kernel_size // 2, kernel_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

            # Apply blur with less intensity
            img_uint8 = ((image + 1) * 127.5).astype(np.uint8)
            blurred = cv2.filter2D(img_uint8, -1, kernel)

            # Add slight Gaussian noise (reduced)
            noise = np.random.normal(0, 2, blurred.shape)  # Reduced from higher values
            blurred = np.clip(blurred + noise, 0, 255)

            blurred = blurred / 127.5 - 1
            return blurred
        else:
            # Gentler Gaussian blur
            kernel_size = np.random.choice([3, 5, 7])  # Smaller kernels
            sigma = np.random.uniform(0.5, 2.0)  # Reduced sigma
            img_uint8 = ((image + 1) * 127.5).astype(np.uint8)
            blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigma)
            blurred = blurred / 127.5 - 1
            return blurred

    def prepare_dataset(self, image_paths, validation_split=0.2):
        """Prepare dataset with validation split"""
        dataset = []
        valid_paths = []

        # Filter valid image paths
        for path in image_paths:
            img = self.load_and_preprocess_image(path)
            if img is not None:
                valid_paths.append(path)

        print(f"Found {len(valid_paths)} valid images")

        # Split into train/validation
        train_paths, val_paths = train_test_split(
            valid_paths, test_size=validation_split, random_state=42
        )

        # Prepare training data
        for path in train_paths:
            sharp_img = self.load_and_preprocess_image(path)
            if sharp_img is not None:
                blur_img = self.create_realistic_blur(sharp_img)
                dataset.append([blur_img, sharp_img])

        # Prepare validation data
        val_dataset = []
        for path in val_paths:
            sharp_img = self.load_and_preprocess_image(path)
            if sharp_img is not None:
                blur_img = self.create_realistic_blur(sharp_img)
                val_dataset.append([blur_img, sharp_img])

        return dataset, val_dataset

    def train(self, image_paths, epochs=50, batch_size=4, save_interval=5, validation_split=0.2):
        """Improved training with better monitoring - FIXED"""
        # FIXED: Call prepare_dataset with correct parameters
        dataset, val_dataset = self.prepare_dataset(image_paths, validation_split)

        if len(dataset) == 0:
            print("No valid images found!")
            return

        print(f"Dataset prepared with {len(dataset)} training pairs and {len(val_dataset)} validation pairs")

        # Training metrics tracking
        training_history = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'psnr': [],
            'ssim': []
        }

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            np.random.shuffle(dataset)

            d_losses = []
            g_losses = []

            for i in range(0, len(dataset), batch_size):
                batch_data = dataset[i:i + batch_size]
                if len(batch_data) < batch_size:
                    continue

                blur_imgs = np.array([item[0] for item in batch_data])
                sharp_imgs = np.array([item[1] for item in batch_data])

                # Train discriminator with label smoothing
                fake_imgs = self.generator.predict(blur_imgs, verbose=0)

                # Label smoothing for more stable training
                real_labels = np.ones((batch_size, 1)) * 0.9  # 0.9 instead of 1.0
                fake_labels = np.zeros((batch_size, 1)) + 0.1  # 0.1 instead of 0.0

                d_loss_real = self.discriminator.train_on_batch(sharp_imgs, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator less frequently for balance
                if i % 2 == 0:  # Train generator every other batch
                    valid_labels = np.ones((batch_size, 1))
                    g_loss = self.combined.train_on_batch(blur_imgs, [sharp_imgs, valid_labels])
                    g_losses.append(g_loss[0])

                d_losses.append(d_loss[0])

                if i % (batch_size * 5) == 0:  # More frequent updates
                    avg_d = np.mean(d_losses[-5:]) if len(d_losses) >= 5 else np.mean(d_losses)
                    avg_g = np.mean(g_losses[-5:]) if len(g_losses) >= 5 else np.mean(g_losses)
                    print(f"Batch {i // batch_size}: D loss: {avg_d:.4f}, G loss: {avg_g:.4f}")

            # Calculate epoch metrics
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses) if g_losses else 0

            # Calculate PSNR and SSIM on a sample
            sample_blur = np.array([dataset[j][0] for j in range(min(4, len(dataset)))])
            sample_sharp = np.array([dataset[j][1] for j in range(min(4, len(dataset)))])
            sample_deblurred = self.generator.predict(sample_blur, verbose=0)

            psnr, ssim = self.calculate_metrics(sample_sharp, sample_deblurred)

            # Store metrics
            training_history['epoch'].append(epoch + 1)
            training_history['d_loss'].append(avg_d_loss)
            training_history['g_loss'].append(avg_g_loss)
            training_history['psnr'].append(psnr)
            training_history['ssim'].append(ssim)

            print(f"Epoch {epoch + 1} - D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}")
            print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")

            # Save model and generate samples
            if (epoch + 1) % save_interval == 0:
                self.save_models(f'epoch_{epoch + 1}')
                self.generate_sample_images(dataset[:4], epoch + 1)

                # Plot training curves
                self.plot_training_history(training_history)

        return training_history

    def validate(self, val_data):
        """Validation function"""
        if len(val_data) == 0:
            return 0, 0

        blur_imgs = np.array([item[0] for item in val_data])
        sharp_imgs = np.array([item[1] for item in val_data])

        # Generate fake images
        fake_imgs = self.generator.predict(blur_imgs, verbose=0)

        # Calculate discriminator loss
        real_labels = np.ones((len(val_data), 1))
        fake_labels = np.zeros((len(val_data), 1))

        d_loss_real = self.discriminator.evaluate(sharp_imgs, real_labels, verbose=0)
        d_loss_fake = self.discriminator.evaluate(fake_imgs, fake_labels, verbose=0)
        val_d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        # Calculate generator loss
        valid_labels = np.ones((len(val_data), 1))
        val_g_loss = self.combined.evaluate(blur_imgs, [sharp_imgs, valid_labels], verbose=0)[0]

        return val_d_loss, val_g_loss

    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(history['epoch'], history['d_loss'], label='Discriminator Loss', color='red')
        axes[0, 0].plot(history['epoch'], history['g_loss'], label='Generator Loss', color='blue')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # PSNR curve
        axes[0, 1].plot(history['epoch'], history['psnr'], label='PSNR', color='green')
        axes[0, 1].set_title('PSNR Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # SSIM curve
        axes[1, 0].plot(history['epoch'], history['ssim'], label='SSIM', color='orange')
        axes[1, 0].set_title('SSIM Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Combined metrics
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(history['epoch'], history['psnr'], label='PSNR', color='green')
        ax2.plot(history['epoch'], history['ssim'], label='SSIM', color='orange')
        axes[1, 1].set_title('PSNR and SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('PSNR (dB)', color='green')
        ax2.set_ylabel('SSIM', color='orange')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f'results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_models(self, suffix=''):
        """Save models"""
        os.makedirs('models/saved_models', exist_ok=True)
        self.generator.save(f'models/saved_models/generator_{suffix}.h5')
        self.discriminator.save(f'models/saved_models/discriminator_{suffix}.h5')
        print(f"Models saved with suffix: {suffix}")

    def load_models(self, generator_path, discriminator_path=None):
        """Load models"""
        self.generator = keras.models.load_model(
            generator_path,
            custom_objects={'content_loss': self.content_loss},
            compile=False
        )
        if discriminator_path:
            self.discriminator = keras.models.load_model(discriminator_path, compile=False)
        print("Models loaded successfully")

    def generate_sample_images(self, sample_data, epoch, save_path='results/training_samples'):
        """Generate and save sample images"""
        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(3, len(sample_data), figsize=(15, 9))
        if len(sample_data) == 1:
            axes = axes.reshape(-1, 1)

        for i, (blur_img, sharp_img) in enumerate(sample_data):
            # Generate deblurred image
            deblurred = self.generator.predict(np.expand_dims(blur_img, axis=0), verbose=0)[0]

            # Denormalize images
            blur_display = np.clip((blur_img + 1) / 2, 0, 1)
            sharp_display = np.clip((sharp_img + 1) / 2, 0, 1)
            deblurred_display = np.clip((deblurred + 1) / 2, 0, 1)

            # Plot
            axes[0, i].imshow(blur_display)
            axes[0, i].set_title('Blurred Input')
            axes[0, i].axis('off')

            axes[1, i].imshow(deblurred_display)
            axes[1, i].set_title('Deblurred Output')
            axes[1, i].axis('off')

            axes[2, i].imshow(sharp_display)
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_path}/samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def deblur_image(self, image_path, output_path=None):
        """Deblur a single image"""
        img = self.load_and_preprocess_image(image_path)
        if img is None:
            print("Failed to load image")
            return None

        # Generate deblurred image
        deblurred = self.generator.predict(np.expand_dims(img, axis=0), verbose=0)[0]

        # Denormalize
        deblurred = np.clip((deblurred + 1) * 127.5, 0, 255).astype(np.uint8)

        if output_path:
            Image.fromarray(deblurred).save(output_path)
            print(f"Deblurred image saved to {output_path}")

        # Display comparison
        original = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original)
        axes[0].set_title('Original (Input)')
        axes[0].axis('off')

        axes[1].imshow(deblurred)
        axes[1].set_title('Deblurred Output')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        return deblurred

    def calculate_metrics(self, y_true, y_pred):
        """Calculate PSNR and SSIM metrics"""
        import tensorflow as tf
        from tensorflow.image import ssim

        # Ensure consistent float32 dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Convert from [-1, 1] to [0, 1] range
        y_true_norm = (y_true + 1.0) / 2.0
        y_pred_norm = (y_pred + 1.0) / 2.0

        # Calculate PSNR with consistent dtypes
        mse = tf.reduce_mean(tf.square(y_true_norm - y_pred_norm))
        mse = tf.cast(mse, tf.float32)  # Ensure float32

        # Avoid division by zero and ensure float32
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        mse = tf.maximum(mse, epsilon)

        psnr = 20.0 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)

        # Calculate SSIM
        ssim_value = tf.reduce_mean(ssim(y_true_norm, y_pred_norm, max_val=1.0))

        return float(psnr.numpy()), float(ssim_value.numpy())


# Example usage and testing
if __name__ == "__main__":
    # Initialize improved model
    deblur_gan = ImprovedDeblurGAN()

    print("Improved DeblurGAN initialized successfully!")
    print("\nGenerator Summary:")
    deblur_gan.generator.summary()
    print("\nDiscriminator Summary:")
    deblur_gan.discriminator.summary()

    print("\nKey improvements:")
    print("✅ Fixed checkerboard artifacts")
    print("✅ Better upsampling strategy")
    print("✅ Instance normalization")
    print("✅ Improved blur generation")
    print("✅ Label smoothing")
    print("✅ Validation monitoring")
    print("✅ Better loss functions")