
# Face Mask Detection using CNN

This repository contains code and a trained Convolutional Neural Network (CNN) model for detecting face masks in images. The model is trained on the "Face Mask Detection Dataset" from Kaggle.

## Dataset

The dataset used for training and testing the face mask detection model is called the "Face Mask Detection Dataset" and can be found on Kaggle. It consists of images of people with and without face masks. The dataset contains two main classes:

1. **WithMask**: Images of people wearing face masks.
2. **WithoutMask**: Images of people without face masks.

The dataset is organized into separate folders for each class, and it is necessary to preprocess the data before training the model.

## Model Architecture

The face mask detection model is built using a Convolutional Neural Network (CNN). The architecture of the model consists of the following layers:

1. **Input Layer**: Accepts input images of size (width, height, channels).
2. **Convolutional Layers**: Consist of multiple convolutional layers with ReLU activation functions to extract relevant features from the input images.
3. **Pooling Layers**: Interspersed with convolutional layers to reduce the spatial dimensions of the feature maps.
4. **Flatten Layer**: Flattens the pooled feature maps into a 1D vector.
5. **Dense Layers**: Fully connected layers with dropout regularization to classify the flattened features into the appropriate classes.
6. **Output Layer**: Generates the final output predictions, representing the probability of the input image belonging to each class.

The model is trained using the dataset mentioned above and achieves good accuracy in detecting face masks in images.

## Requirements

To run the code and train the face mask detection model, you need the following dependencies:

- Python (3.x)
- TensorFlow (2.x)
- Keras
- NumPy
- Matplotlib

You can install the required packages by running the following command:

```shell
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```shell
git clone https://github.com/AniketSurjuse/face_mask_detectionCNN.git
cd face-mask-detection
```

2. Download the "Face Mask Detection Dataset" from Kaggle and extract it into the project directory.

3. Preprocess the dataset:

- Split the dataset into training and testing sets.
- Perform any necessary data augmentation techniques.

4. Train the model:

```shell
python train.py
```

5. Evaluate the model:

```shell
python evaluate.py
```

6. Test the model on new images:

```shell
python detect.py --image /path/to/image.jpg
```

## Results

The face mask detection model achieves an accuracy of 92% on the test set. It performs well in distinguishing between images of people wearing face masks and those without face masks.

## Conclusion

The trained CNN model can be used effectively for face mask detection in images. It can be further improved by tuning hyperparameters, increasing the dataset size, or using more advanced architectures. Feel free to experiment and contribute to this project!
