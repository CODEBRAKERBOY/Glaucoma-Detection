import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt  # Added import for visualization

# Paths to image and mask directories
IMAGE_PATH = r'C:\Users\alok\Desktop\REFUGE2\train\images'
MASK_PATH = r'C:\Users\alok\Desktop\REFUGE2\train\mask'

# Parameters
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3  # Adjust based on the number of classes in your mask

def load_image(image_path):
    """Load and resize image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to load image at {image_path}")
        return None
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def load_mask(mask_path):
    """Load and resize mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Unable to load mask at {mask_path}")
        return None
    mask = cv2.resize(mask, IMAGE_SIZE)
    mask = mask / 255.0  # Normalize to [0, 1]
    mask = to_categorical(mask, num_classes=NUM_CLASSES)  # Convert to one-hot encoding
    return mask

def process_single_image_and_mask(image_dir, mask_dir):
    # Get the first image and mask file
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.bmp')]
    
    if not image_files or not mask_files:
        raise ValueError("No image or mask files found in the specified directories.")
    
    img_file = image_files[0]
    msk_file = mask_files[0]
    
    img_path = os.path.join(image_dir, img_file)
    msk_path = os.path.join(mask_dir, msk_file)
    
    img = load_image(img_path)
    msk = load_mask(msk_path)
    
    if img is not None and msk is not None:
        # Debug outputs
        print(f"Processed {img_file}: Image shape {img.shape}")
        print(f"Processed {msk_file}: Mask shape {msk.shape}")
    else:
        print("Error processing the image or mask.")
    
    return img, msk

def display_image_and_mask(img, mask):
    """Display the image and mask."""
    plt.figure(figsize=(10, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis('off')
    
    # Display mask
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(np.argmax(mask, axis=-1))  # Show mask as categorical labels
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    img, msk = process_single_image_and_mask(IMAGE_PATH, MASK_PATH)
    if img is not None and msk is not None:
        display_image_and_mask(img,msk)
