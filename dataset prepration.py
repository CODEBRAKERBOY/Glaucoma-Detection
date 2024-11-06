# Paths for storing converted images
train_converted_path = '/content/drive/My Drive/DATASET/train_converted_images'
val_converted_path = '/content/drive/My Drive/DATASET/val_converted_images'
test_converted_path = '/content/drive/My Drive/DATASET/test_converted_images'

# Create necessary directories for each class (0 and 1) in train, validation, and test
for path in [train_converted_path, val_converted_path, test_converted_path]:
    for class_dir in ['0', '1']:
        dir_path = os.path.join(path, class_dir)
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path)
