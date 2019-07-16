#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a simple RNN/LSTM model using keras, where cnews dataset is used.

"""
import logging
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

logging.getLogger().setLevel(logging.INFO)


def load_data(train_data_path, valid_data_path, test_data_path):
    """Load train, validate and test dataset.

    Args:
        train_data_path: string, the path pf train dataset,
        valid_data_path: string, the path pf validate dataset,
        test_data_path: string, the path pf test dataset.

    Returns:
        train_dataset: DataFrame, the train dataset,
        valid_dataset: DataFrame, the validate dataset,
        test_dataset: DataFrame, the test dataset.
    """

    train_dataset = pd.read_csv(train_data_path)
    train_dataset = train_dataset.take(np.random.permutation(len(train_dataset))[:5000])
    logging.info("Read train dataset successfully")

    valid_dataset = pd.read_csv(valid_data_path)
    logging.info("Read validate dataset successfully")

    test_dataset = pd.read_csv(test_data_path)
    logging.info("Read test dataset successfully")

    return train_dataset, valid_dataset, test_dataset


def label_preprocess(train_dataset, valid_dataset, test_dataset):
    """Preprocess the labels of train, validate and test data.

    Args:
        train_dataset: DataFrame, the train dataset,
        valid_dataset: DataFrame, the validate dataset,
        test_dataset: DataFrame, the test dataset.

    Returns:
        Y_train: array, shape (n_samples, n_labels), ont-hot encoded labels of train dataset,
        Y_valid: array, shape (n_samples, n_labels), ont-hot encoded labels of validate dataset,
        Y_test: array, shape (n_samples, n_labels), ont-hot encoded labels of test dataset,
    """

    # Get labels of train, validate, test dataset and reshape to (n_samples, 1)
    train_labels = train_dataset.label.values.reshape(-1, 1)
    valid_labels = valid_dataset.label.values.reshape(-1, 1)
    test_labels = test_dataset.label.values.reshape(-1, 1)

    # label_encoder = LabelEncoder()
    # train_labels = label_encoder.fit_transform(train_labels).reshape(-1, 1)
    # valid_labels = label_encoder.transform(valid_labels).reshape(-1, 1)
    # test_labels = label_encoder.transform(test_labels).reshape(-1, 1)

    # One-hot encode labels of train, validate, test dataset
    onehot_encoder = OneHotEncoder()
    Y_train = onehot_encoder.fit_transform(train_labels).toarray()
    Y_valid = onehot_encoder.transform(valid_labels).toarray()
    Y_test = onehot_encoder.transform(test_labels).toarray()

    return Y_train, Y_valid, Y_test


def text_prepreocess(train_dataset, valid_dataset, test_dataset, max_words, max_len):
    """Preprocess the texts of train, validate and test data.

    Args:
        train_dataset: DataFrame, the train dataset,
        valid_dataset: DataFrame, the validate dataset,
        test_dataset: DataFrame, the test dataset,
        max_words: int, the maximal number of words that are used for tokenizing,
        max_len: int, the maximal length of each sequence.

    Returns:
        train_padded_sequences: nested list, the padded sequences of train dataset,
        valid_padded_sequences: nested list, the padded sequences of validate dataset,
        test_padded_sequences: nested list, the padded sequences of test dataset.
    """

    # Get the texts data
    train_text = train_dataset.cutword
    valid_text = valid_dataset.cutword
    test_text = test_dataset.cutword

    # Initialize a tokenizer and fit on the texts of train dataset
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_text)

    # Convert original text to a sequence =, i.e., vectoring
    train_sequences = tokenizer.texts_to_sequences(train_text)
    valid_sequences = tokenizer.texts_to_sequences(valid_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)

    # Pad each sequence to the same length, i.e., max_len
    train_padded_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len)
    valid_padded_sequences = keras.preprocessing.sequence.pad_sequences(valid_sequences, maxlen=max_len)
    test_padded_sequences = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len)

    return train_padded_sequences, valid_padded_sequences, test_padded_sequences


def rnn_model(input_dim, input_length):
    """Build RNN model.

    Args:
        input_dim: int, the vocabulary size,
        input_length: int, the length of input sequence.

    Returns:
        The RNN model
    """

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
        keras.layers.SimpleRNN(units=128, dropout=0.2, return_sequences=True),
        keras.layers.SimpleRNN(units=128, dropout=0.2),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=10, activation="softmax")
    ])
    logging.info("Build RNN model successfully.")

    # Compole the RNN model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    logging.info("Compile RNN model successfully.")

    return model


def lstm_model(input_dim, input_length):
    """Build LSTM model.

    Args:
        input_dim: int, the vocabulary size,
        input_length: int, the length of input sequence.

    Returns:
        The LSTM model
    """

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
        keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True),
        keras.layers.LSTM(units=128, dropout=0.2),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=10, activation="softmax")
    ])
    logging.info("Build LSTM model successfully.")

    # Compole the LSTM model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    logging.info("Compile LSTM model successfully.")

    return model


def train_model(model, X_train, Y_train, X_valid, Y_valid, epochs, batch_size):
    """Train the given model.

    Args:
        model: the build model
        X_train: nested list, the padded sequences of train dataset,
        Y_train: array, shape (n_samples, n_labels), ont-hot encoded labels of train dataset,
        X_valid: nested list, the padded sequences of validate dataset,
        Y_valid: array, shape (n_samples, n_labels), ont-hot encoded labels of validate dataset,
        epochs: int, the number of epochs,
        batch_size: int, the size of batch.

    Returns:
        the trained model.
    """

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))
    logging.info("Train the given model successfully.")
    print("loss: {}, accuracy: {}".format(history.history["loss"], history.history["acc"]))

    # Initialize a figure
    fig, ax = plt.subplots(2, 1)

    # Plot loss curve
    ax[0].plot(list(range(1, epochs+1)), history.history["loss"])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].set_title("Loss curve during the training process")

    # Plot accuracy curve
    ax[1].plot(list(range(1, epochs+1)), history.history["acc"])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy curve during the training process")

    plt.show()

    return model


def test_model(trained_model, X_test, Y_test):
    """Test the given model.

    Args:
        trained_model: the trained given model,
        X_test: nested list, the padded sequences of test dataset,
        Y_test: array, shape (n_samples, n_labels), ont-hot encoded labels of test dataset.

    Returns:
        None.
    """

    # Test the trained model
    score = trained_model.evaluate(X_test, Y_test, verbose=0)
    logging.info("Test the given model successfully.")

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Get the predict results of test data, Y_test_predict is a nested list, shape (n_samples, n_lables)
    Y_test_predict = trained_model.predict(X_test)

    # Get the confusion matrix of predict result
    confu_matrix = confusion_matrix(np.argmax(Y_test_predict, axis=1), np.argmax(Y_test, axis=1))

    # Initialize a figure
    plt.figure(figsize=(8, 8))

    # Plot the heatmap of confusion matrix
    sns.heatmap(confu_matrix.T, square=True, annot=True, fmt='d', cbar=False, linewidths=.8)
    plt.xlabel('True label', size=14)
    plt.ylabel('Predicted label', size=14)
    label_names = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
    plt.xticks(np.arange(10) + 0.5, label_names, size=12)
    plt.yticks(np.arange(10) + 0.3, label_names, size=12)
    plt.show()


if __name__ == "__main__":

    # Set data paths
    train_data_path = "data/cnews_train.csv"
    valid_data_path = "data/cnews_val.csv"
    test_data_path = "data/cnews_test.csv"

    # Set maximal words for using, maximal length of sequence, the number of epochs and the size of batch
    max_words = 5000
    max_len = 600
    epochs = 10
    batch_size = 128

    # Load train, validate, test dataset
    train_dataset, valid_dataset, test_dataset = load_data(train_data_path, valid_data_path, test_data_path)

    # Preprocess labels and texts of each dataset
    Y_train, Y_valid, Y_test = label_preprocess(train_dataset, valid_dataset, test_dataset)
    train_padded_sequence, valid_padded_sequence, test_padded_sequence = text_prepreocess(train_dataset, valid_dataset, test_dataset, max_words, max_len)

    # Build a RNN model and LSTM model
    rnn_model = rnn_model(max_words, max_len)
    lstm_model = lstm_model(max_words, max_len)

    # Train the RNN or LSTM model on train dataset, and then test it on the test dataset
    train_model = train_model(rnn_model, train_padded_sequence, Y_train, valid_padded_sequence, Y_valid, epochs, batch_size)
    test_model(train_model, test_padded_sequence, Y_test)
