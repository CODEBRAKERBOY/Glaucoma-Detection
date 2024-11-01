# Glaucoma Detection Using Deep Learning

## Problem Statement
Glaucoma is a chronic eye disease that damages the optic nerve, potentially leading to vision loss or blindness if not detected early. Traditional diagnostic methods require skilled professionals and can be costly and time-consuming, making it difficult to implement at a large scale. Given the global prevalence of glaucoma, there is an urgent need for accessible and accurate automated diagnostic tools.

This project aims to develop a deep learning-based solution for glaucoma detection using fundus images of the eye. By training a neural network on labeled images, the model can classify images as either **Glaucoma Positive** or **Glaucoma Negative**, helping in early detection and diagnosis. The goal is to create a model that performs well across various datasets and minimizes false positives and false negatives, thereby assisting healthcare providers in screening and decision-making.

The project leverages ResNet18, a convolutional neural network architecture, for binary classification of glaucoma, with the following objectives:
- **High Accuracy**: Maximize overall classification accuracy.
- **Low False Positives**: Reduce misclassification of healthy eyes as glaucoma-positive.
- **Low False Negatives**: Avoid missing cases where glaucoma is present.

This solution, once optimized, could assist in developing automated systems for early glaucoma screening, benefiting both clinicians and patients by enabling prompt and affordable diagnostics.

<br>
<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-EE4C2B?style=for-the-badge&logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/deep%20learning-000000?style=for-the-badge&logo=ai&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003B57?style=for-the-badge&logo=matplotlib&logoColor=white)

</div>
<br>

## Required Modules

To run this project, you need to install the following Python packages. You can install them using pip:

- `torch`
- `torchvision`
- `torchaudio`
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `pandas`

## Additional Dependencies

Make sure you have the following modules for optimal performance:

- **CUDA** (if using a GPU): Ensure your system supports CUDA for faster training if you have a compatible NVIDIA GPU.
- **Jupyter Notebook** (optional): For interactive experimentation and visualization.

## Dataset Structure
The dataset should be structured as follows:

DATASET
├── train/
│   ├── Glaucoma_Positive/
│   └── Glaucoma_Negative/
└── val/
    ├── Glaucoma_Positive/
    └── Glaucoma_Negative/



