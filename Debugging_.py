import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import random

# Function to load images and labels from a directory
def load_images_and_labels(image_directory):
    images = []
    labels = []
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for root, dirs, files in os.walk(image_directory):
        for file in files:
            image_path = os.path.join(root, file)

            if os.path.splitext(file)[1].lower() in valid_image_extensions:
                try:
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        images.append(np.array(img))

                    # Assign label based on directory name
                    if '1' in root:
                        labels.append(1)
                    elif '0' in root:
                        labels.append(0)

                except Exception as e:
                    print(f"Failed to load image {image_path}: {e}")
            else:
                print(f"Skipping non-image file: {image_path}")

    return images, labels

# Function to visualize a sample of images and their labels
def visualize_samples(images, labels, num_samples=5):
    if len(images) < num_samples:
        print(f"Not enough images to display. Available: {len(images)}")
        num_samples = len(images)

    class_0_indices = [i for i, label in enumerate(labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(labels) if label == 1]

    sampled_indices = random.sample(class_0_indices, min(num_samples // 2, len(class_0_indices))) + \
                      random.sample(class_1_indices, min(num_samples // 2, len(class_1_indices)))

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sampled_indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Label: {labels[idx]}")
        plt.axis('off')
    plt.show()

# Function to check and print class distribution
def check_class_distribution(labels):
    counter = Counter(labels)
    print("Class distribution:", dict(counter))

# Function to visualize label distribution
def visualize_label_distribution(labels):
    counter = Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts, color=['blue', 'orange'])
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Class Distribution')
    plt.xticks(classes, ['Glaucoma Negative (0)', 'Glaucoma Positive (1)'])
    plt.show()

if __name__ == "__main__":
    # Set your training directory here
    train_dir = '/content/drive/My Drive/DATASET/train'

    # Load images and labels
    print("Loading and verifying training images...")
    train_images, train_labels = load_images_and_labels(train_dir)

    # Check and print class distribution
    check_class_distribution(train_labels)

    # Visualize label distribution
    print("Visualizing label distribution...")
    visualize_label_distribution(train_labels)

    # Visualize some samples from the training set
    print("Visualizing a sample of training images...")
    visualize_samples(train_images, train_labels)

    # Print the number of images loaded
    print(f"Loaded {len(train_images)} images with corresponding labels.")
