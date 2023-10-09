# Cat and Dog Image Classification with Deep Learning

This repository contains code for training a deep learning model to classify images of cats and dogs. The model is trained on the Kaggle Cats and Dogs dataset, which you can find [here](https://www.kaggle.com/datasets/tongpython/cat-and-dog). The goal is to create a model that can distinguish between images of cats and dogs with high accuracy.

## Dataset

The dataset used for this project is the Kaggle Cats and Dogs dataset. It contains a large collection of labeled images of cats and dogs for training and testing the model.
You can download the dataset from [this link](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

## Code Overview

### Prerequisites

Before running the code, you'll need to install the required libraries and have Python and PyTorch set up on your machine. You can install the necessary packages by running:

```bash
pip install -r requirements.txt
```

### Data Preprocessing
The code includes functions for loading and preprocessing the dataset.
Images are resized to 100x100 pixels, converted to RGB format, and augmented with various transformations to improve model performance.

### Model Architecture
The deep learning model used in this project is a Convolutional Neural Network (CNN).
It consists of multiple convolutional layers, batch normalization, max-pooling, dropout, and fully connected layers. The model architecture is defined in the CNN class within the code.

### Training and Evaluation
The model is trained using the training set and evaluated on the test set.
 The code includes a training loop that calculates loss and accuracy during training. Evaluation metrics, such as loss and accuracy, are displayed and plotted to assess the model's performance.

### Predicting Images
You can use the trained model to make predictions on new images of cats and dogs.
The pred_and_show function allows you to provide an image file path, preprocess the image, and display the prediction result along with the image.

## Usage
1.Clone this repository:
```bash
git clone https://github.com/DaniPopov/cat-dog-classification.git
cd cat-dog-classification
```
2. Download the Kaggle Cats and Dogs dataset and place it in the ./data directory.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Kaggle for providing the Cats and Dogs dataset.

Medium artical by [Muhammad Ardi](https://github.com/MuhammadArdiPutra) "Cat Dog Classification with CNN" 
which you can find [here](https://python.plainenglish.io/cat-dog-classification-with-cnn-84af3ae98c44).


Feel free to contribute, open issues, or provide feedback to improve this project!






