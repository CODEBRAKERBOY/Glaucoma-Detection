import cv2
import os
import matplotlib.pyplot as plt

# Load and display a sample image with green channel extraction and CLAHE
def display_sample_image(image_path):
    # Load and convert the image to RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract green channel and apply CLAHE
    green_channel = image_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0)
    enhanced_image = clahe.apply(green_channel)

    # Plot original, green channel, and CLAHE-enhanced images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    axs[0].imshow(image_rgb)
    axs[0].set_title('Original Image', color='red')
    axs[0].axis('off')
    axs[1].imshow(green_channel, cmap='gray')
    axs[1].set_title('Green Channel', color='green')
    axs[1].axis('off')
    axs[2].imshow(enhanced_image, cmap='gray')
    axs[2].set_title('CLAHE Enhanced Image', color='blue')
    axs[2].axis('off')
    plt.show()

# Process all images in a directory, applying CLAHE, and save to target directory
def process_and_save_images(main_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    images = sorted(os.listdir(main_path))
    for image_name in images:
        image_path = os.path.join(main_path, image_name)
        image = cv2.imread(image_path)[:, :, 1]  # Load and extract green channel
        clahe = cv2.createCLAHE(clipLimit=2.0)
        processed_image = clahe.apply(image)
        cv2.imwrite(os.path.join(target_path, image_name), processed_image)

# Example usage
# display_sample_image('/content/drive/My Drive/DATASET/train/0/sjchoi86-HRF-243.png')
# process_and_save_images('/content/drive/My Drive/DATASET/train/0', '/content/drive/My Drive/DATASET/train_converted_images/0')
