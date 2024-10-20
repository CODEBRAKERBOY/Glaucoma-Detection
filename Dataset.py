import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_and_labels(image_directory):
    """
    Load images from the specified directory and assign labels.
    Assumes that the folder names indicate the classes (1: Glaucoma Present, 0: Glaucoma not Present).
    """
    images = []
    labels = []
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}  # Valid extensions for images

    # Walk through all files in the image directory
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            image_path = os.path.join(root, file)

            # Check if the file is an image by extension
            if os.path.splitext(file)[1].lower() in valid_image_extensions:
                try:
                    # Load and convert image to RGB
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        images.append(np.array(img))

                    # Assign label based on the directory name
                    if '1' in root:  # 1: Glaucoma Present
                        labels.append(1)
                    elif '0' in root:  # 0: Glaucoma not Present
                        labels.append(0)

                except Exception as e:
                    print(f"Failed to load image {image_path}: {e}")
            else:
                print(f"Skipping non-image file: {image_path}")

    return images, labels

def preprocess_images(images, target_size=(224, 224)):
    """
    Resize and normalize images.
    Images are normalized to [0, 1] range and resized to the target size.
    """
    resized_images = []

    for img in images:
        # Resize image to the target size
        img_resized = np.array(Image.fromarray(img).resize(target_size))
        resized_images.append(img_resized)

        # Print shapes for debugging
        print(f"Image shape after resize: {img_resized.shape}")

    images_normalized = np.array(resized_images) / 255.0  # Normalize images to [0, 1]

    return images_normalized

def create_image_generators(train_dir, val_dir, test_dir, target_size=(224, 224), batch_size=32):
    """
    Create image generators for train, validation, and test datasets.
    """
    # Augmentation for training set to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow images in batches from directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator, test_generator

def main():
    # Define paths for images in your dataset
    train_dir = '/content/drive/My Drive/DATASET/train'
    val_dir = '/content/drive/My Drive/DATASET/val'
    test_dir = '/content/drive/My Drive/DATASET/test'

    print("Setting up image generators...")
    train_gen, val_gen, test_gen = create_image_generators(train_dir, val_dir, test_dir)

    # Print out the class indices for debugging
    print(f"Class indices: {train_gen.class_indices}")

    # Training set example
    print(f"Training data: {train_gen.samples} images")
    print(f"Validation data: {val_gen.samples} images")
    print(f"Test data: {test_gen.samples} images")

if __name__ == "__main__":
    main()

