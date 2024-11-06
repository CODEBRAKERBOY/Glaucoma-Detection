import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Assuming 'test_take1_' is your test dataset, 'pred' contains model predictions,
# 'test_df['label']' contains the true labels, and 'classes' maps class indices to class names.

def random_test_sample_with_prediction(SEED):
    idxs = np.random.default_rng(seed=SEED).permutation(len(pred))[:5]
    batch_idx = idxs // BATCH_SIZE
    image_idx = idxs - batch_idx * BATCH_SIZE
    idx = idxs

    fig, axs = plt.subplots(1, 5, figsize=(12, 12), dpi=150)

    for i in range(5):
        img = test_take1_[batch_idx[i]][0][image_idx[i]]
        label = test_take1_[batch_idx[i]][1][image_idx[i]].numpy()

        if int(pred[idx[i]]) == label:
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
            axs[i].set_title('image (no: ' + str(idx[i]) + ')\n' + classes[label], fontsize=8, color='green')
        else:
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
            axs[i].set_title('image (no: ' + str(idx[i]) + ')\n' + classes[label], fontsize=8, color='red')  # False prediction

# Generate random samples and display predictions
random_test_sample_with_prediction(SEED=14)
random_test_sample_with_prediction(SEED=53)
random_test_sample_with_prediction(SEED=674)
random_test_sample_with_prediction(SEED=9)

# After all the predictions are made, compute and display the classification report
print("--" * 50)
print("Classification Report for the Test Set:")

# Assuming 'test_df['label']' contains the true labels and 'pred' contains model predictions
print(classification_report(test_df['label'], pred, target_names=list(classes.values())))
