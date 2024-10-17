import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_and_masks(image_directory, mask_directory):
    """
    Load images and corresponding masks from specified directories.
    Assumes that images and masks have the same filenames, allowing for different extensions.
    """
    images = []
    masks = []
    labels = []
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}  # Valid extensions for images and masks

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

                    # Find the corresponding mask file (same name, different extension possible)
                    mask_name = os.path.splitext(file)[0]  # Get the name without extension
                    mask_file = None

                    # Remove spaces and handle parentheses in filenames
                    mask_name_clean = mask_name.replace(" ", "")  # Remove spaces
                    print(f"Searching for mask: {mask_name_clean}")

                    # Search for the mask file in the mask directory
                    for ext in valid_image_extensions:
                        potential_mask_path = os.path.join(mask_directory, mask_name_clean + ext)
                        print(f"Checking: {potential_mask_path}")
                        
                        if os.path.exists(potential_mask_path):
                            mask_file = potential_mask_path
                            break

                    if mask_file:
                        with Image.open(mask_file) as mask:
                            mask = mask.convert('L')  # 'L' mode for grayscale
                            masks.append(np.array(mask))
                    else:
                        print(f"Warning: No matching mask found for image {file} in {mask_directory}")
                        print(f"Image Path: {image_path}, Mask Path: {mask_name_clean}")
                        continue

                    # Example label assignment, assuming binary classification (adjust if multi-class)
                    label = 1 if 'glaucoma' in file.lower() else 0  # Example condition for labels
                    labels.append(label)

                except Exception as e:
                    print(f"Failed to load image or mask {image_path}: {e}")
            else:
                print(f"Skipping non-image file: {image_path}")

    return images, masks, labels

def preprocess_images_and_masks(images, masks, target_size=(224, 224)):
    """
    Resize and normalize images and masks.
    Masks are normalized to [0, 1] range and resized to the target size.
    """
    resized_images = []
    resized_masks = []

    for img, mask in zip(images, masks):
        # Resize image and mask to the target size
        img_resized = np.array(Image.fromarray(img).resize(target_size))
        mask_resized = np.array(Image.fromarray(mask).resize(target_size, Image.NEAREST))  # Use nearest-neighbor for masks

        resized_images.append(img_resized)
        resized_masks.append(mask_resized)

        # Print shapes for debugging
        print(f"Image shape after resize: {img_resized.shape}")
        print(f"Mask shape after resize: {mask_resized.shape}")

    images_normalized = np.array(resized_images) / 255.0
    masks_normalized = np.array(resized_masks) / 255.0  # Normalize masks

    # Ensure masks have the correct shape (height, width, 1)
    masks_normalized = np.expand_dims(masks_normalized, axis=-1)

    return images_normalized, masks_normalized

def main():
    # Define paths for images and masks in REFUGE2 dataset on your Google Drive
    test_image_dir = '/content/drive/My Drive/REFUGE2/test/images/dummy_class'
    test_mask_dir = '/content/drive/My Drive/REFUGE2/test/mask/dummy_class'

    print("Processing images and masks...")
    images, masks, labels = load_images_and_masks(test_image_dir, test_mask_dir)

    # Check if any images or masks were loaded
    if len(images) == 0 or len(masks) == 0:
        raise ValueError("No images or masks were loaded. Check the dataset path and file formats.")

    # Preprocess images and masks
    images, masks = preprocess_images_and_masks(images, masks)

    # Split the data into training, validation, and test sets
    X_train_img, X_temp_img, X_train_mask, X_temp_mask, y_train, y_temp = train_test_split(
        images, masks, labels, test_size=0.4, random_state=42)

    X_val_img, X_test_img, X_val_mask, X_test_mask, y_val, y_test = train_test_split(
        X_temp_img, X_temp_mask, y_temp, test_size=0.5, random_state=42)

    # Print the shape of the datasets
    print(f"Training set size: {X_train_img.shape}, Masks size: {X_train_mask.shape}, Labels size: {len(y_train)}")
    print(f"Validation set size: {X_val_img.shape}, Masks size: {X_val_mask.shape}, Labels size: {len(y_val)}")
    print(f"Test set size: {X_test_img.shape}, Masks size: {X_test_mask.shape}, Labels size: {len(y_test)}")

if __name__ == "__main__":
    main()
