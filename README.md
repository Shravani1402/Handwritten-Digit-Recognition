# Handwritten Digit Recognition
 This is the *Handwritten Digit Recognition* Deep Learning Project.
 First, We have downloaded the Mnist Dataset from Kaggle.
 The MNIST dataset contains 60,000 training images of handwritten digits from zero to nine and 10,000 images for testing. So, the MNIST dataset has 10 different classes.The handwritten digits images are represented as a 28×28 matrix where each cell contains grayscale pixel value. 
keras is imported as the high-level neural networks API. This is part of the TensorFlow library and provides a convenient interface for building and training neural networks.
You import necessary modules and functions from Keras for working with neural networks:
Sequential: A linear stack of layers for building a neural network model layer by layer.
Dense: A fully connected layer that implements the operation: output = activation(dot(input, kernel) + bias).
Dropout: A regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
Flatten: Flattens the input, useful when transitioning from convolutional layers to dense layers.
Conv2D: Convolutional layer for 2D spatial convolution over images.
MaxPooling2D: Max pooling operation for spatial data.
mnist.load_data() loads the MNIST dataset and returns two tuples containing the training and testing data:
-: x_train, y_train: Training images and their corresponding labels.
-: x_test, y_test: Testing images and their corresponding labels.
print(x_train.shape, y_train.shape) prints the shape of the training data (x_train) and labels (y_train). It gives you an idea of the number of samples, image dimensions, and label dimensions in the training set.





