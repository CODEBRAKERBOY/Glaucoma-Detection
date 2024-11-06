import pandas as pd
from sklearn.utils import shuffle

# Function to list image paths in a directory
def create_images_list(path):
    return [os.path.join(path, img) for img in sorted(os.listdir(path))]

# Collect image paths for train, validation, and test sets
train_data_0 = create_images_list(os.path.join(train_converted_path, '0'))
train_data_1 = create_images_list(os.path.join(train_converted_path, '1'))
val_data_0 = create_images_list(os.path.join(val_converted_path, '0'))
val_data_1 = create_images_list(os.path.join(val_converted_path, '1'))
test_data_0 = create_images_list(os.path.join(test_converted_path, '0'))
test_data_1 = create_images_list(os.path.join(test_converted_path, '1'))

# Create DataFrames for train, validation, and test sets
train_df = pd.concat([
    pd.DataFrame({'image': train_data_0, 'label': 0}),
    pd.DataFrame({'image': train_data_1, 'label': 1})
], ignore_index=True)

val_df = pd.concat([
    pd.DataFrame({'image': val_data_0, 'label': 0}),
    pd.DataFrame({'image': val_data_1, 'label': 1})
], ignore_index=True)

test_df = pd.concat([
    pd.DataFrame({'image': test_data_0, 'label': 0}),
    pd.DataFrame({'image': test_data_1, 'label': 1})
], ignore_index=True)

# Shuffle the DataFrames
SEED = 42  # Seed for reproducibility
train_df = shuffle(train_df, random_state=SEED).reset_index(drop=True)
val_df = shuffle(val_df, random_state=SEED).reset_index(drop=True)
test_df = shuffle(test_df, random_state=SEED).reset_index(drop=True)

# Print the shape of the DataFrames for verification
print("Train images -> ", train_df.shape[0])
print("Validation images -> ", val_df.shape[0])
print("Test images -> ", test_df.shape[0])
