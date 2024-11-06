import matplotlib.pyplot as plt

# Plotting Training and Validation Loss and Accuracy
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

# Plotting Loss
axs[0].grid(linestyle='--', alpha=0.7)
axs[0].plot(hist.history['loss'], label='Training Loss', color='blue')
axs[0].plot(hist.history['val_loss'], label='Validation Loss', color='orange')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epochs', fontsize=10)
axs[0].set_ylabel('Loss', fontsize=10)
axs[0].legend(fontsize=10)

# Plotting Accuracy
axs[1].grid(linestyle='--', alpha=0.7)
axs[1].plot(hist.history['acc'], label='Training Accuracy', color='blue')
axs[1].plot(hist.history['val_acc'], label='Validation Accuracy', color='orange')
axs[1].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('Epochs', fontsize=10)
axs[1].set_ylabel('Accuracy', fontsize=10)
axs[1].legend(fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()
