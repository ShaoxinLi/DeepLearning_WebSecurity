#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a simple CNN model using tensorflow, where the cifar-10 dataset is used.

"""
import numpy as np
import pickle
import tensorflow as tf


def unpickle(file_path):
    """Unpickle raw data to a dict.

    Args:
        file_path: string, the path of cifar-10 batch data.

    Returns:
        dict, its structure is as follows:
            data: array, shape (n_samples, n_pixels),
            labels: list, the labels of images,
            batch_label: string, the name of the current batch,
            filename: list, the name of each image.
    """

    with open(file_path, "rb") as binary_file:

        batch_data = pickle.load(binary_file, encoding="latin1")

    return batch_data


def load_data(batch_data_paths):
    """Load cifar-10 images data and labels data.

    Args:
        batch_data_paths: list, the paths of data batch.

    Returns:
        array, shape (n_samples, n_pixels), two-dimensional array of images data,
        array, shape (n_samples, ), two-dimensional array of labels data.
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


def cnn(X_train, y_train, X_test, y_test, model_path):
    """Build CNN model, train and test on cifar-10 dataset.

    Args:
        X_train: array, shape (n_images, 32, 32, 3), the images data for training model.
        y_train: array, shape (n_labels), the labels data for training model.
        X_test: array, shape (n_images, 32, 32, 3), the images data for testing model.
        y_test: array, shape (n_labels), the labels data for testing model.
        model_path: string, the path to save trained CNN model.

    Returns:
        None
    """

    num_classes = len(set(y_train))
    num_epoch = 10

    # Set placeholder for X, y, and dropout rate
    data_placeholder = tf.placeholder(tf.float32, [None, X_train.shape[1], X_train.shape[2], 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])
    dropout_placeholdr = tf.placeholder(tf.float32)

    # Build CNN model, which consists of two convolutional layer and one dense layer
    conv0 = tf.layers.conv2d(data_placeholder, filters=20, kernel_size=5, activation=tf.nn.relu)
    pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2], strides=[2, 2])
    conv1 = tf.layers.conv2d(inputs=pool0, filters=40, kernel_size=4, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])
    flatten = tf.layers.flatten(pool1)
    dense = tf.layers.dense(inputs=flatten, units=400, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropout_placeholdr)
    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    # The predicted labels is the label with highest probability
    y_predicted = tf.argmax(input=logits, axis=1)

    # One hot encode the real labels, and use mean cross entropy loss.
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels_placeholder, num_classes), logits=logits)
    mean_loss = tf.reduce_mean(losses)

    # Use adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

    # Define a saver for save and load model
    saver = tf.train.Saver()

    # Star a session
    with tf.Session() as sess:

        print("Training the CNN model...")
        sess.run(tf.global_variables_initializer())

        # Input train data and dropout rate to model
        train_feed_dict = {data_placeholder: X_train, labels_placeholder: y_train, dropout_placeholdr: 0.25}

        for epoch in range(num_epoch):

            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
            print("Epoch = {}/{}, Mean loss = {}".format(epoch, num_epoch, mean_loss_val))

        # Save trained model to model_path
        saver.save(sess, model_path)

        print("Testing the CNN model...")

        # Load trained model from model_path
        saver.restore(sess, model_path)

        # Input test data to model
        test_feed_dict = {data_placeholder: X_test, labels_placeholder: y_test, dropout_placeholdr: 0}

        # Get the predicted label for each test sample and mean cross entropy loss on test dataset
        y_predicted, mean_loss = sess.run([y_predicted, mean_loss], feed_dict=test_feed_dict)

        print('Test loss', mean_loss)
        print('Test accuracy', tf.metrics.accuracy(y_test, y_predicted))


if __name__ == "__main__":

    model_path = "model/cnn_model"

    batch_train_data_paths = [
        "data/cifar-10/data_batch_1",
        "data/cifar-10/data_batch_2",
        "data/cifar-10/data_batch_3",
        "data/cifar-10/data_batch_4",
        "data/cifar-10/data_batch_5"
    ]

    batch_test_data_paths = ["data/cifar-10/test_batch"]

    # Load cifar-10 data for traning and testing
    image_data_train, image_labels_train = load_data(batch_train_data_paths)
    image_data_test, image_labels_test = load_data(batch_test_data_paths)

    cnn(image_data_train, image_labels_train, image_data_test, image_labels_test, model_path)



