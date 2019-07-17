#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This is a demo of building classifier on mnist dataset, where KNN, SVM, DNN and CNN models are used for comparision.

"""
import os
import struct
import logging
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger().setLevel(logging.INFO)


def load_data(file_path):
    """Load mnist images data, read binary file as numpy array.

    Args:
        file_path: string, the path of mnist images data.

    Returns:
        images_data: array, shape (n_images, n_pixels), two-dimensional array of images data.
    """

    with open(file_path, 'rb') as binary_file:

        buffers = binary_file.read()
        logging.info("Read image binary file successfully.")

        # Read the first 4 integer numbers of the image file
        magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
        logging.info("Unpack image binary file successfully.")

        # The whole images data bits
        bits = num * rows * cols

        # Read images data
        images_data = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
        logging.info("Read images data successfully.")

    # Reshape three-dimensional images data to two-dimensional
    images_data = np.reshape(images_data, [num, rows * cols])

    return images_data


def load_labels(file_path):
    """Load mnist labels data, read binary file as numpy array.

    Args:
        file_path: string, the path of mnist labels data.

    Returns:
        labels_data: array, shape (n_labels), one-dimensional array of labels data.
    """

    with open(file_path, 'rb') as binary_file:

        buffers = binary_file.read()
        logging.info("Read label binary file successfully.")

        # Read the first 2 integer numbers of the label file
        magic, num = struct.unpack_from('>II', buffers, 0)
        logging.info("Unpack label binary file successfully.")

        # Read labels data
        labels_data = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
        logging.info("Read labels data successfully.")

    # Reshape two-dimensional labels data to one-dimensional
    labels_data = np.reshape(labels_data, [num])

    return labels_data


def dnn_model():
    """Build a DNN model.

    Returns:
        The DNN model.
    """

    # DNN model structure
    model = keras.models.Sequential([
        keras.layers.Dense(units=512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    logging.info("Build DNN model successfully")

    # Compile the DNN model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    logging.info("Compile DNN model successfully")

    return model


def cnn_model():
    """Build a CNN model.

    Returns:
        The CNN model.
    """

    # CNN model structure
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(28, 28, 1), padding="valid", activation="relu", kernel_initializer="uniform"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu", kernel_initializer="uniform"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation="tanh"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=256, activation="tanh"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=10, activation="softmax")
    ])
    logging.info("Build CNN model successfully")

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info("Compile CNN model successfully")

    return model


def train_model(model, X_train, y_train, epochs, batch_size, input_flag, output_flag, name):
    """Train the model

    Args:
        model: the built model,
        X_train: array, shape (n_samples, n_pixels), the images train data,
        y_train: array, shape (n_samples), the labels train data,
        epochs: int, the number of epoch,
        batch_size: int, the size of each batch,
        input_flag: string, indicate the type of the input,
        output_flag: string, indicate the type of the output.

    Returns:
        the trained model and the number of classes.
    """

    num_classes = len(set(y_train))

    # For traditional ML model, directly fit
    if input_flag == "vector" and output_flag == "scalar":

        X_train = X_train.astype('float32') / 255

        logging.info("Begin training the {} model".format(name))
        model.fit(X_train, y_train)
        logging.info("Training {} model successfully".format(name))

    # For the neural network that outputs a vector (such as DNN), fit after convert class vectors to binary class matrices
    if input_flag == "vector" and output_flag == "vector":

        X_train = X_train.astype('float32') / 255
        Y_train = keras.utils.to_categorical(y_train, num_classes)

        logging.info("Begin training the {} model".format(name))
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        logging.info("Training {} model successfully".format(name))

    # For the neural network that inputa a tensor (such as CNN), fit after reshape image data as a tensor
    if input_flag == "tensor":

        X_train = np.reshape(X_train, [X_train.shape[0], 28, 28, 1])
        Y_train = keras.utils.to_categorical(y_train, num_classes)

        logging.info("Begin training the {} model".format(name))
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        logging.info("Training {} model successfully".format(name))

    return model, num_classes


def test_model(trained_model, X_test, y_test, num_classes, input_flag, output_flag, name):
    """Test the trained model.

    Args:
        trained_model: the trained model
        X_test: array, shape (n_samples, n_pixels), the images test data,
        y_test: array, shape (n_samples), the labels test data,
        num_classes: int, the number of classes,
        input_flag: string, indicate the type of the input,
        output_flag: string, indicate the type of the output.

    Returns:
        acc_score: float, the test accuracy.
    """

    # For traditional ML model, directly predict
    if input_flag == "vector" and output_flag == "scalar":

        X_test = X_test.astype('float32') / 255

        logging.info("Begin testing the {} model".format(name))
        y_test_predict = trained_model.predict(X_test)
        logging.info("Tesing {} model successfully".format(name))

        acc_score = accuracy_score(y_test, y_test_predict)

    # For the neural network that outputs a vector (such as DNN), predict after convert class vectors to binary class matrices
    if input_flag == "vector" and output_flag == "vector":

        X_test = X_test.astype('float32') / 255
        Y_test = keras.utils.to_categorical(y_test, num_classes)

        logging.info("Begin testing the {} model".format(name))
        socre = trained_model.evaluate(X_test, Y_test, verbose=1)
        logging.info("Tesing {} model successfully".format(name))

        acc_score = socre[1]

    # For the neural network that inputa a tensor (such as CNN), predict after reshape image data as a tensor
    if input_flag == "tensor":

        X_test = np.reshape(X_test, [X_test.shape[0], 28, 28, 1])
        Y_test = keras.utils.to_categorical(y_test, num_classes)

        logging.info("Begin testing the {} model".format(name))
        socre = trained_model.evaluate(X_test, Y_test, verbose=1)
        logging.info("Tesing {} model successfully".format(name))

        acc_score = socre[1]

    return acc_score


if __name__ == "__main__":

    # Load mnist data for traning and testing
    train_images = load_data('../dnn_mnist_classification/data/train-images-idx3-ubyte')
    train_labels = load_labels('../dnn_mnist_classification/data/train-labels-idx1-ubyte')
    test_images = load_data('../dnn_mnist_classification/data/t10k-images-idx3-ubyte')
    test_labels = load_labels('../dnn_mnist_classification/data/t10k-labels-idx1-ubyte')

    # Set the number of epoch and the size of batch
    epochs = 10
    batch_size = 128

    # Get classification model
    knn_model = KNeighborsClassifier(n_neighbors=15, algorithm="auto", n_jobs=-1)
    svm_model = SVC(C=1.0, decision_function_shape="ovo")
    dnn_model = dnn_model()
    cnn_model = cnn_model()

    # Pack built models and their related information
    models = [knn_model, svm_model, dnn_model, cnn_model]
    names = ["KNN", "SVM", "DNN", "CNN"]
    input_flags = ["vector", "vector", "vector", "tensor"]
    output_flags = ["scalar", "scalar", "vector", "vector"]

    # Store the final test accuracy scores
    accuracy_scores = {}

    # Train and test on each model in models
    for (model, name, input_flag, output_flag) in zip(models, names, input_flags, output_flags):

        trained_model, num_classes = train_model(model, train_images, train_labels, epochs, batch_size, input_flag, output_flag, name)
        acc_score = test_model(model, test_images, test_labels, num_classes, input_flag, output_flag, name)
        accuracy_scores[name] = acc_score

    print(accuracy_scores)







