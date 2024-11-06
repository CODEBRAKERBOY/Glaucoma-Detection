import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Assuming 'test_df' is your DataFrame containing the true labels
# and 'classes' is a dictionary mapping class indices to their names

# Validation and Test evaluations of ViT model
with stg.scope():  # Scoped block for model evaluation and predictions
    print('ViT model results')
    print('--' * 50)

    # Evaluate on validation dataset
    val_eval_vit = model.evaluate(val_dataset)
    print('Validation Loss: {0:.3f}'.format(val_eval_vit[0]))
    print('Validation Accuracy: {0:.3f} %'.format(val_eval_vit[1] * 100))  # Predictions

with stg.scope():  # Scoped block for test predictions
    # Make predictions on the test dataset
    pred = model.predict(test_dataset)
    pred = np.argmax(pred, axis=1)  # Assuming a classification task (e.g., multi-class classification)

# Predictions and scores calculation
mse = mean_squared_error(test_df['label'], pred)
f1 = f1_score(test_df['label'], pred, average='weighted')
acc = accuracy_score(test_df['label'], pred)

# Output evaluation metrics
print('Mean Squared Error : {0:.5f}'.format(mse))
print('Weighted F1 Score : {0:.3f}'.format(f1))
print('Accuracy Score : {0:.3f} %'.format(acc * 100))
print('--' * 50)

# Test evaluation (Final test set evaluation)
test_eval_vit = model.evaluate(test_dataset)
print('Test Loss: {0:.3f}'.format(test_eval_vit[0]))
print('Test Accuracy: {0:.3f} %'.format(test_eval_vit[1] * 100))

# Print classification report for more detailed evaluation per class
clf = classification_report(test_df['label'], pred, target_names=list(classes.values()))
print(clf)

# Confusion Matrix Visualization
cm = confusion_matrix(test_df['label'], pred)

# Create confusion matrix display
cmd = ConfusionMatrixDisplay(cm, display_labels=list(classes.values()))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))  # Increased figure size for better readability
cmd.plot(ax=ax, cmap='Purples', colorbar=False)
plt.title("Confusion Matrix")
plt.show()
