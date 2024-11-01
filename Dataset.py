import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
from sklearn.utils import class_weight
import traceback

# Mount Google Drive
drive.mount('/content/drive')

# Updated paths for training from Google Drive
train_dir = '/content/drive/My Drive/DATASET/train'
val_dir = '/content/drive/My Drive/DATASET/val'
test_dir = '/content/drive/My Drive/DATASET/test'

def load_images_and_labels(image_directory):
    """
    Load images from the specified directory and assign labels.
    Assumes that the folder names indicate the classes:
    '0' for class 0 (Glaucoma_Negative) and '1' for class 1 (Glaucoma_Positive).
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
                    if '1' in root:  # Glaucoma Present (1)
                        labels.append(1)
                    elif '0' in root:  # Glaucoma Not Present (0)
                        labels.append(0)

                except Exception as e:
                    print(f"Failed to load image {image_path}: {e}")
                    traceback.print_exc()
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
        # Resize image to the target size with high-quality interpolation
        img_resized = np.array(Image.fromarray(img).resize(target_size, Image.ANTIALIAS))
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

    # Flow images in batches from directories (seed added here)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',  # Use 'binary' since you have two classes
        classes=['1', '0'],   # Ensure class names are strings as per your directory structure
        seed=42  # Set seed here for reproducibility
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['1', '0'],   # Ensure class names are strings as per your directory structure
        seed=42  # Set seed here for reproducibility
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['1', '0']    # Ensure class names are strings as per your directory structure
    )

    return train_generator, val_generator, test_generator

def calculate_class_weights(generator):
    """
    Calculate class weights based on the generator's class distribution.
    """
    class_counts = np.bincount(generator.classes)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    return dict(enumerate(class_weights))

if __name__ == "__main__":
    # Instantiate the image generators
    print("Creating image generators...")
    train_gen, val_gen, test_gen = create_image_generators(train_dir, val_dir, test_dir)

    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    print("Class Weights:", class_weights)

    # Print out class indices for verification
    print(f"Class indices (train): {train_gen.class_indices}")
    print(f"Number of training samples: {train_gen.samples}")
    print(f"Number of validation samples: {val_gen.samples}")
    print(f"Number of test samples: {test_gen.samples}")


