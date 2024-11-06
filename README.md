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
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![React Router](https://img.shields.io/badge/React_Router-CA4245?style=for-the-badge&logo=react-router&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%238511FA.svg?style=for-the-badge&logo=bootstrap&logoColor=white)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

</div>
<br>

## Required Modules

## Required Modules

- **numpy**: Array manipulation and numerical operations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities like metrics and preprocessing

### Deep Learning and Image Processing
- **tensorflow**: Deep learning framework; includes Keras for model building
- **keras**: Interface for the TensorFlow deep learning framework (optional if using TensorFlow 2.x)
- **opencv-python**: Image processing library (optional, for image transformations)
- **Pillow**: Image processing library, used with Keras for loading images

### Visualization
- **matplotlib**: Plotting and visualization of data and model metrics
- **seaborn**: Statistical data visualization (optional, enhances matplotlib)


You can install the modules individually with pip install, or add them to a requirements.txt file:

- `pandas`
- `scikit-learn`
- `tensorflow`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `numpy`

  * Start by opening Google Colab.

* Ensure that you have access to a TPU by selecting `Runtime` > `Change runtime type` > `Hardware accelerator` > `TPU`.

* Clone the repository using the following command:

    ```bash
    !git clone https://github.com/CODEBRAKERBOYY/GlaucomaDetection.git
    ```

* Change into the cloned directory:

    ```bash
    %cd GlaucomaDetection
    ```

* Install the required modules by running:

    ```bash
    !pip install -r requirements.txt
    ```

* To train your model or make predictions, run the following command:

    ```bash
    !python main.py [arguments]
    ```

    **Arguments:**
    - `train_model` - Specify this if you want to train the model before inference.
    - `existing` - Use this after the `train_model` argument if you want to retrain an existing model.
    - `make_predictions` / `None` - This loads the existing model for inference.


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

    * While giving path as input always remember it consists only tow class one is Negative and other is Positive.
* Perform image preprocesssing if you think it's necessary otherwise skip it.
* Path for data images if you wants to retrain model:
* https://drive.google.com/drive/folders/1M89d5jKBInbhvmEC95zn51zD6A25HKbF?usp=share_link

  ### Accuracy plot

![Accuracy Plot](https://github.com/CODEBRAKERBOYY/Glaucoma-Detection/blob/main/assets/Unknown-13.png)

  ### Image Processing
  





