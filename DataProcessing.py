# Data Processing Pipeline

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = (224, 224)

# Load and preprocess the image
image_path = '/content/drive/My Drive/DATASET/train/0/sjchoi86-HRF-243.png'  # Update with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency in display

# Extract Green Channel
green_channel = image_rgb[:, :, 1]

# Apply CLAHE to the Green Channel
clahe = cv2.createCLAHE(clipLimit=2.0)
clahe_img = clahe.apply(green_channel)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image', color='red')
axs[0].axis('off')
axs[1].imshow(green_channel, cmap='gray')
axs[1].set_title('Green Channel', color='green')
axs[1].axis('off')
axs[2].imshow(clahe_img, cmap='gray')
axs[2].set_title('After CLAHE', color='blue')
axs[2].axis('off')

plt.show()
