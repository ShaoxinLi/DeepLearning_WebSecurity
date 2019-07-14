#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a simple DNN model using keras, where the mnist dataset is used.

"""
import struct
import numpy as np
from tensorflow import keras


def load_data(file_path):
    """Load mnist images data, read binary file as numpy array.

    Args:
        file_path: string, the path of mnist images data.

    Returns:
        array, shape (n_images, n_pixels), two-dimensional array of images data.
    """

    with open(file_path, 'rb') as binary_file:

        buffers = binary_file.read()

        # Read the first 4 integer numbers of the image file
        magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)

        # The whole images data bits
        bits = num * rows * cols

        # Read images data
        images_data = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))

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

        # Read the first 2 integer numbers of the label file
        magic, num = struct.unpack_from('>II', buffers, 0)

        # Read labels data
        labels_data = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))

    # Reshape two-dimensional labels data to one-dimensional
    labels_data = np.reshape(labels_data, [num])

    return labels_data


def dnn(X_train, y_train, X_test, y_test):
    """Build a DNN model, train the model with mnist train dataset, and finally test the model on the mnist test dataset.

    Args:
        X_train: array, shape (n_images, n_pixels), the images data for training model.
        y_train: array, shape (n_labels), the labels data for training model.
        X_test: array, shape (n_images, n_pixels), the images data for testing model.
        y_test: array, shape (n_labels), the labels data for testing model.

    Returns:
        None
    """

    batch_size = 128
    num_classes = 10
    epochs = 20

    X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
    X_train, X_test = X_train / 255,  X_test / 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(y_train, num_classes)
    Y_test = keras.utils.to_categorical(y_test, num_classes)

    # Build DNN model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=512, activation='relu', input_shape=(784, )))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=512, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test loss', score[0])
    print('Test accuracy', score[1])


if __name__ == '__main__':

    # Load mnist data for traning and testing
    train_images = load_data('data//train-images-idx3-ubyte')
    train_labels = load_labels('data/train-labels-idx1-ubyte')
    test_images = load_data('data/t10k-images-idx3-ubyte')
    test_labels = load_labels('data/t10k-labels-idx1-ubyte')

    dnn(train_images, train_labels, test_images, test_labels)


