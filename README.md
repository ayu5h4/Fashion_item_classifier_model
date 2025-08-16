# Fashion Item Classifier Model

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying fashion items from the FashionMNIST dataset. The model is based on the TinyVGG architecture and is designed to serve as a practical example of computer vision for multi-class classification. With 10 layers

## Description

The project walks through the process of building, training, and evaluating a deep learning model to identify ten different types of fashion apparel from grayscale images. It starts with a basic linear model and progressively builds up to a more complex CNN architecture to achieve higher accuracy. This repository is a great resource for those looking to get started with computer vision and PyTorch.

## Dataset

The model is trained on the **FashionMNIST** dataset, which is a popular alternative to the original MNIST dataset. It consists of:

  * 60,000 training images
  * 10,000 testing images
  * Each image is a 28x28 grayscale image.
  * There are 10 classes of fashion items.

The 10 classes are:

1.  T-shirt/top
2.  Trouser
3.  Pullover
4.  Dress
5.  Coat
6.  Sandal
7.  Shirt
8.  Sneaker
9.  Bag
10. Ankle boot

## Model Architecture

The final model (`FashionMNISTv2`) is a Convolutional Neural Network inspired by the TinyVGG architecture. It consists of two convolutional blocks followed by a classifier.

  * **Convolutional Block 1:**
      * `nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1, stride=1)`
      * `nn.ReLU()`
      * `nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, stride=1)`
      * `nn.ReLU()`
      * `nn.MaxPool2d(kernel_size=2, stride=2)`
  * **Convolutional Block 2:**
      * `nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)`
      * `nn.ReLU()`
      * `nn.MaxPool2d(kernel_size=2, stride=2)`
  * **Classifier:**
      * `nn.Flatten()`
      * `nn.Linear(in_features=490, out_features=10)`

## Getting Started

### Prerequisites

  * Python 3.x
  * PyTorch
  * TorchVision
  * Matplotlib
  * Requests
  * tqdm

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your_username/Fashion_item_classifier_model.git
    ```
2.  Install the required packages:
    ```bash
    pip install torch torchvision matplotlib requests tqdm
    ```

## Usage

The primary code for this project is in the `FashionMNIST.ipynb` Jupyter Notebook. You can open and run this notebook to see the entire process of data loading, model creation, training, and evaluation.

### Training

The model can be trained by running the training loop cells in the notebook. The training process uses the following hyperparameters:

  * **Epochs:** 5
  * **Batch Size:** 32
  * **Optimizer:** Stochastic Gradient Descent (SGD) with a learning rate of 0.075
  * **Loss Function:** Cross-Entropy Loss

### Evaluation

The notebook includes a function `eval_model` to evaluate the performance of the trained model on the test dataset.

## Results

The CNN model (`FashionMNISTv2`) demonstrates a significant improvement in performance over the baseline linear models. The training and testing loss progressively decrease over the epochs, indicating that the model is learning effectively. For detailed results and visualizations, please refer to the `FashionMNIST.ipynb` notebook.
