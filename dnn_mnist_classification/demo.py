#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a simple DNN model using keras, where the mnist dataset is used.

"""
import struct
import numpy as np
from tensorflow import keras
import logging
logging.getLogger().setLevel(logging.INFO)


def load_data(file_path):
    """Load mnist images data, read binary file as numpy array.

    Args:
        file_path: string, the path of mnist images data.

    Returns:
        array, shape (n_images, n_pixels), two-dimensional array of images data.
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
        array, shape (n_labels), one-dimensional array of labels data.
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
    logging.info("Build DNN model successfully.")

    # Compole the DNN model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    logging.info("Compile DNN model successfully.")

    return model


def train_model(model, X_train, y_train, epochs, batch_size):
    """Train the builded DNN model on mnist training data.

    Args:
        model: the DNN model/
        X_train: array, shape (n_images, n_pixels), the images data for training model.
        y_train: array, shape (n_labels), the labels data for training model.
        epochs: int, the number of epoch.
        batch_size: int, the size of each batch.

    Returns:
        The trained DNN model and the number of classes of mnist dataset.
    """

    # Convert train data to float type
    X_train = X_train.astype('float32') / 255
    print(X_train.shape[0], 'train samples')

    # Convert class vectors to binary class matrices
    num_classes = len(set(y_train))
    Y_train = keras.utils.to_categorical(y_train, num_classes)

    # Train the DNN model
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    logging.info("Train DNN model successfully.")
    print("loss: {}, accuracy: {}".format(history.history["loss"], history.history["acc"]))

    return model, num_classes


def test_model(trained_model, X_test, y_test, num_classes):
    """Test the trained DNN model on mnist test dataset.

    Args:
        trained_model: the trained DNN model.
        X_test: array, shape (n_images, n_pixels), the images data for testing model.
        y_test: array, shape (n_labels), the labels data for testing model.
        num_classes: int, the number of classes of mnist dataset.

    Returns:
        None
    """

    # Convert test data to float type
    X_test = X_test.astype('float32') / 255
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    Y_test = keras.utils.to_categorical(y_test, num_classes)

    # Test the trained DNN model
    score = trained_model.evaluate(X_test, Y_test, verbose=0)
    logging.info("Test DNN model successfully.")

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':

    # Load mnist data for traning and testing
    train_images = load_data('data//train-images-idx3-ubyte')
    train_labels = load_labels('data/train-labels-idx1-ubyte')
    test_images = load_data('data/t10k-images-idx3-ubyte')
    test_labels = load_labels('data/t10k-labels-idx1-ubyte')

    # Set the number of epoch and the size of each batch
    epochs = 10
    batch_size = 128

    # Get DNN model
    model = dnn_model()

    # Train the DNN model on mnist train dataset
    trained_model, num_classes = train_model(model, train_images, train_labels, epochs, batch_size)

    # Test the trained DNN model on mnist test dataset
    test_model(trained_model, test_images, test_labels, num_classes)



