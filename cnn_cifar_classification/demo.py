#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a simple CNN model using tensorflow, where the cifar-10 dataset is used.

"""
import os
import random
import pickle
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger().setLevel(logging.INFO)


def unpickle(file_path):
    """Unpickle raw data to a dict.

    Args:
        file_path: string, the path of cifar-10 batch data.

    Returns:
        batch_data: dict, its structure is as follows:
            data: array, shape (n_samples, n_pixels),
            labels: list, the labels of images,
            batch_label: string, the name of the current batch,
            filename: list, the name of each image.
    """

    with open(file_path, "rb") as binary_file:

        batch_data = pickle.load(binary_file, encoding="latin1")
        logging.info("Unpickle binary file successfully")

    return batch_data


def load_data(batch_data_paths):
    """Load cifar-10 images data and labels data.

    Args:
        batch_data_paths: list, the paths of data batch.

    Returns:
        image_data: array, shape (n_samples, 32, 32, 3), four-dimensional array of images data,
        image_labels: list, the labels of samples.
    """

    # Store images data and labels data of each batch
    image_data_list = []
    image_labels_list = []

    for path in batch_data_paths:

        # Get images data and labels data of each batch
        image_data_batch = unpickle(path)["data"]
        image_labels_batch = unpickle(path)["labels"]

        image_data_list.append(image_data_batch)
        image_labels_list.append(image_labels_batch)

    # Concatenate the images data of each batch and reshape to (n_samples, 32, 32, 3)
    image_data = np.concatenate(image_data_list, axis=0)
    image_data = np.reshape(image_data, [image_data.shape[0], 32, 32, 3])

    # Flatten image_labels_list
    image_labels = [label for image_labels_batch in image_labels_list for label in image_labels_batch]

    return image_data, image_labels


def shuffle_data(image_data, image_labels):
    """Shuffle the dataset

    Args:
        image_data: array, shape (n_images, 32, 32, 3), image data for training,
        image_labels: list, image labels for training

    Returns:
        image_data_shuffled: array, shape (n_images, 32, 32, 3), shuffled image data,
        image_labels_shuffled: array, shape (n_images), shuffled image labels.
    """

    row_indices = list(range(image_data.shape[0]))
    random.shuffle(row_indices)

    image_data_shuffled = image_data[row_indices]
    image_labels_shuffled = np.array(image_labels)[row_indices]

    return image_data_shuffled, image_labels_shuffled


def get_batch(image_data, image_labels, batch_size, cur_iteration, total_iteration):
    """Get batch of dataset

    Args:
        image_data: array, shape (n_images, 32, 32, 3), the image data for training,
        image_labels: array, shape (n_images), the image labels for training,
        batch_size: int, the size of each batch,
        cur_iteration: int, the current iteration number,
        total_iteration: int, the total number of iteration.

    Returns:
        image_data_batch: array, shape (batch_size, 32, 32, 3), a batch of image data,
        image_labels_batch: array, shape (batch_size), a batch of image labels.
    """

    if cur_iteration < total_iteration-1:

        image_data_batch = image_data[cur_iteration*batch_size: (cur_iteration+1)*batch_size]
        image_labels_batch = image_labels[cur_iteration*batch_size: (cur_iteration+1)*batch_size]
    else:

        image_data_batch = image_data[cur_iteration*batch_size:]
        image_labels_batch = image_labels[cur_iteration*batch_size:]

    return image_data_batch, image_labels_batch


def cnn_model(data_placeholder, labels_placeholder, dropout_placeholdr, num_classes):
    """Build CNN model.

    Args:
        data_placeholder: placeholder of traning data,
        labels_placeholder: placeholder of training labels,
        dropout_placeholdr: placeholder of dropout rate,
        num_classes: the number of classes of cifar-10 dataset.

    Returns:
        the logits, optimizer of CNN model and the mean loss.
    """

    # Build CNN model, which consists of two convolutional layer and one dense layer
    conv0 = tf.layers.conv2d(data_placeholder, filters=20, kernel_size=5, activation=tf.nn.relu)
    pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2], strides=[2, 2])
    conv1 = tf.layers.conv2d(inputs=pool0, filters=40, kernel_size=4, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])
    flatten = tf.layers.flatten(pool1)
    dense = tf.layers.dense(inputs=flatten, units=400, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropout_placeholdr)
    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    # One hot encode the real labels, and use mean cross entropy loss.
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels_placeholder, num_classes), logits=logits)
    mean_loss = tf.reduce_mean(losses)

    # Use adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)
    logging.info("Build CNN model successfully")

    return logits, optimizer, mean_loss


def train_model(data_placeholder, labels_placeholder, dropout_placeholdr, optimizer, mean_loss, X_train, y_train, epochs, batch_size, model_path):
    """ Train CNN model on cifar-10 train dataset.

    Args:
        data_placeholder: placeholder of traning data,
        labels_placeholder: placeholder of training labels,
        dropout_placeholdr: placeholder of dropout rate,
        optimizer: optimizer of DNN model,
        mean_loss: mean_loss of DNN model,
        X_train: array, shape (n_images, 32, 32, 3), the images data for training model,
        y_train: list, the labels data for training model,
        epochs: int, the number of epochs,
        model_path: the path to save trained CNN model.

    Returns:
        None
    """

    # Define a saver for save and load model
    saver = tf.train.Saver()

    # Star a session
    with tf.Session() as sess:

        # Initiate global variables
        sess.run(tf.global_variables_initializer())

        # Record the loss in each iteration
        losses = []

        # Get the number of total iteration
        total_iteration = X_train.shape[0] // batch_size

        for epoch in range(epochs):

            # Shuffle dataset in each epoch
            X_train, y_train = shuffle_data(X_train, y_train)
            logging.info("Shuffle dataset successfully")

            # Train CNN model
            for iteration in range(total_iteration):

                # Get a batch data from the whole training dataset
                X_train_batch, y_train_batch = get_batch(X_train, y_train, 1000, iteration, total_iteration)
                logging.info("Get batch data successfully")

                # Input batch data and set dropout rate
                train_feed_dict = {data_placeholder: X_train_batch, labels_placeholder: y_train_batch, dropout_placeholdr: 0.25}

                # Training the CNN model with a batch of data and record losses
                _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
                logging.info("Train the CNN model successfully")
                losses.append(mean_loss_val)

                print("Epoch = {}/{}, Iteration = {}/{}, Mean loss = {}".format(epoch+1, epochs, iteration, total_iteration, mean_loss_val))

        # Initialize a figure
        fig, ax = plt.subplots()

        # Plot loss curve
        ax.plot(list(range(1, epochs*total_iteration+1)), losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cross Entropy Loss")
        ax.set_title("Loss curve during the training process")
        plt.show()

        # Save trained model to model path
        saver.save(sess, model_path)
        logging.info("Save the trained model successfully")


def test_model(data_placeholder, labels_placeholder, dropout_placeholdr, logits, mean_loss, X_test, y_test, model_path):
    """

    Args:
        data_placeholder: placeholder of traning data,
        labels_placeholder: placeholder of training labels,
        dropout_placeholdr: placeholder of dropout rate,
        X_test: array, shape (n_images, 32, 32, 3), the images data for testing model,
        y_test: list, the labels data for testing model,
        model_path: the path to load trained CNN model.

    Returns:
        None
    """

    # Define a saver for save and load model
    saver = tf.train.Saver()

    # Star a session
    with tf.Session() as sess:

        saver.restore(sess, model_path)
        logging.info("Load the trained CNN model successfully")

        # Input test data to model
        test_feed_dict = {data_placeholder: X_test, labels_placeholder: y_test, dropout_placeholdr: 0}

        # The predicted labels is the label with highest probability
        y_predicted = tf.argmax(input=logits, axis=1)

        # Get the predicted label for each test sample and mean cross entropy loss on test dataset
        y_predicted, mean_loss = sess.run([y_predicted, mean_loss], feed_dict=test_feed_dict)
        logging.info("Successfully predict the test samples")

        print('Test loss', mean_loss)
        print('Test accuracy', tf.metrics.accuracy(y_test, y_predicted))


if __name__ == "__main__":

    # Set the save path of trained model, the number of epoch and the size of each batch
    model_path = "model/cnn_model"
    epochs = 10
    batch_size = 1000

    # Set the paths of train and test data
    batch_train_data_paths = [
        "data/data_batch_1",
        "data/data_batch_2",
        "data/data_batch_3",
        "data/data_batch_4",
        "data/data_batch_5"
    ]

    batch_test_data_paths = ["data/test_batch"]

    # Load cifar-10 data for train and testing
    image_data_train, image_labels_train = load_data(batch_train_data_paths)
    image_data_test, image_labels_test = load_data(batch_test_data_paths)

    # Compute the number of classes of cifar-10 dataset
    num_classes = len(set(image_labels_train))

    # Set placeholder for X, y, and dropout rate
    data_placeholder = tf.placeholder(tf.float32, [None, image_data_train.shape[1], image_data_train.shape[2], 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])
    dropout_placeholdr = tf.placeholder(tf.float32)

    # Build CNN model
    logits, optimizer, mean_loss = cnn_model(data_placeholder, labels_placeholder, dropout_placeholdr, num_classes)

    # Train CNN model on cifar-10 train dataset
    train_model(data_placeholder, labels_placeholder, dropout_placeholdr, optimizer, mean_loss, image_data_train,
                image_labels_train, epochs, batch_size, model_path)

    # Test CNN model on cifar-10 test dataset
    test_model(data_placeholder, labels_placeholder, dropout_placeholdr, logits, mean_loss, image_data_test,
               image_labels_test, model_path)




