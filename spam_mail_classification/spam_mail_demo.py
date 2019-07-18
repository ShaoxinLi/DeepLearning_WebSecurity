#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This a demo of spam mail classification, where enron-spam dataset is used.

"""
import os
import logging
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

logging.getLogger().setLevel(logging.INFO)


def load_one_file(file_path):
    """Load single file.

    Args:
        file_path: string, the path of file

    Returns:
        text: string, the content of this file
    """

    text = ""

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

        logging.info("Load mail file {} successfully".format(file_path))

        for line in file:

            # Remove \n and \r in each line
            line = line.strip('\n')
            line = line.strip('\r')
            text += line

        return text


def load_files_from_dir(dir_path):
    """Load files from a directory.

    Args:
        dir_path: string, the path of directory

    Returns:
        text_list: list, store texts of the directory
    """

    text_list = []
    file_list = os.listdir(dir_path)

    for i in range(len(file_list)):

        # Get the file path
        file_path = os.path.join(dir_path, file_list[i])

        if os.path.isfile(file_path):

            logging.info("File path is legal: {}".format(file_path))

            # Load the text content of the file
            text = load_one_file(file_path)
            text_list.append(text)

    return text_list


def load_all_files():
    """Load all files

    Returns:
        ham_texts: list, store all ham mail texts of enron-spam dataset
        spam_texts: list, store all spam mail texts of enron-spam dataset
    """

    ham_texts = []
    spam_texts = []

    for i in range(1, 7):

        ham_path = "data/enron{}/ham".format(str(i))
        logging.info("Load ham directory: {}".format(ham_path))

        # Add the ham texts of the current directory
        ham_texts += load_files_from_dir(ham_path)

        spam_path = "data/enron{}/spam".format(str(i))
        logging.info("Load spam directory: {}".format(spam_path))

        # Add the spam texts of the current directory
        spam_texts += load_files_from_dir(spam_path)

    return ham_texts, spam_texts


def get_binary_vector(ham_texts, spam_texts, max_words):
    """Get the binary vector of each text

    Args:
        ham_texts: list, store all ham mail texts of enron-spam dataset
        spam_texts: list, store all spam mail texts of enron-spam dataset
        max_words: the maximal number of words that are used

    Returns:
        X: array, shape (n_samples, max_words), the binary vectorized text data
        y: array, shape (n_samples), the labels of dataset
    """

    # Get all texts of enron-spam dataset
    texts = ham_texts + spam_texts

    # Initialize a binary vectorizer
    binary_vectorizer = CountVectorizer(decode_error="ignore", lowercase=True, stop_words="english",
                                        max_features=max_words, binary=True)

    # Transform each text of X into a binary vector
    X = binary_vectorizer.fit_transform(texts).toarray()
    logging.info("Transform text into binary vector successfully")

    # Construct labels array y
    y = [0] * len(ham_texts) + [1] * len(spam_texts)
    y = np.asarray(y)

    return X, y


def get_tf_vector(ham_texts, spam_texts, max_words):
    """Get the tf vector of each text

    Args:
        ham_texts: list, store all ham mail texts of enron-spam dataset
        spam_texts: list, store all spam mail texts of enron-spam dataset
        max_words: the maximal number of words that are used

    Returns:
        X: array, shape (n_samples, max_words), the tf vectorized text data
        y: array, shape (n_samples), the labels of dataset
    """

    # Get all texts of enron-spam dataset
    texts = ham_texts + spam_texts

    # Initialize a tf vectorizer
    counter_vectorizer = CountVectorizer(decode_error="ignore", lowercase=True, stop_words="english",
                                        max_features=max_words, binary=False)

    # Transform each text of X into a tf vector
    X = counter_vectorizer.fit_transform(texts)
    X = X.toarray()
    logging.info("Transform text into tf vector successfully")

    # Construct labels array y
    y = [0] * len(ham_texts) + [1] * len(spam_texts)
    y = np.asarray(y)

    return X, y


def get_tfidf_vector(ham_texts, spam_texts, max_words):
    """Get the tf-idf vector of each text

    Args:
        ham_texts: list, store all ham mail texts of enron-spam dataset
        spam_texts: list, store all spam mail texts of enron-spam dataset
        max_words: the maximal number of words that are used

    Returns:
        X: array, shape (n_samples, max_words), the tf-idf vectorized text data
        y: array, shape (n_samples), the labels of dataset
    """

    # Get all texts of enron-spam dataset
    texts = ham_texts + spam_texts

    # Initialize a tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer(decode_error="ignore", lowercase=True, stop_words="english",
                                         max_features=max_words, binary=False)

    # Transform each text of X into a tf-idf vector
    X = tfidf_vectorizer.fit_transform(texts).toarray()
    logging.info("Transform text into tf-idf vector successfully")

    # Construct labels array y
    y = [0] * len(ham_texts) + [1] * len(spam_texts)
    y = np.asarray(y)

    return X, y


def get_vocabulary_vector(ham_texts, spam_texts, max_words, max_length):
    """Get the vocabulary vector of each text

    Args:
        ham_texts: list, store all ham mail texts of enron-spam dataset
        spam_texts: list, store all spam mail texts of enron-spam dataset
        max_words: int, the maximal number of words that are used
        max_length: int, the length of each sequence

    Returns:
        X: array, shape (n_samples, max_length), the tf-idf vectorized text data
        y: array, shape (n_samples), the labels of dataset
    """

    # Get all texts of enron-spam dataset
    texts = ham_texts + spam_texts

    # Initialize a tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)

    # Fit the tokenizer on all texts
    tokenizer.fit_on_texts(texts)

    # Transform each text of X into a sequence, which is padded to max_length
    X = tokenizer.texts_to_sequences(texts)
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=max_length)
    logging.info("Transform text into vocabulary vector successfully")

    # Construct labels array y
    y = [0] * len(ham_texts) + [1] * len(spam_texts)
    y = np.asarray(y)

    return X, y


def dnn(vector_length):
    """Build DNN model

    Args:
        vector_length: int, the length of input vector

    Returns:
        the built dnn model
    """

    # DNN model structure
    model = keras.models.Sequential([
        keras.layers.Dense(units=512, activation='relu', input_shape=(vector_length,)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    logging.info("Build DNN model successfully")

    # Compile the DNN model
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    logging.info("Compile DNN model successfully")

    return model


def cnn(input_dim, input_length):
    """Build CNN model

    Args:
        input_dim: int, the number of words in the vocabulary
        input_length: int, the length of sequence

    Returns:
        the built CNN model
    """

    # CNN model structure
    model = keras.models.Sequential([
        # Embedding layer is necessary for process text data
        keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
        keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="valid", activation="relu", kernel_initializer="uniform"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="valid", activation="relu", kernel_initializer="uniform"),
        keras.layers.MaxPooling1D(pool_size=2),
        # There must be a flatten layer because the output of the last layer is still a tensor
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation="tanh"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=256, activation="tanh"),
        keras.layers.Dropout(rate=0.2),
        # Sigmoid is used for binary classification
        keras.layers.Dense(units=1, activation="sigmoid")
    ])
    logging.info("Build CNN model successfully")

    # Compile the CNN model
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info("Compile CNN model successfully")

    return model


def rnn(input_dim, input_length):
    """Build RNN model.

    Args:
        input_dim: int, the number of words in the vocabulary
        input_length: int, the length of input sequence

    Returns:
        The built RNN model
    """

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
        keras.layers.SimpleRNN(units=128, dropout=0.2, return_sequences=True),
        keras.layers.SimpleRNN(units=128, dropout=0.2),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(rate=0.2),
        # Sigmoid is used for binary classification
        keras.layers.Dense(units=1, activation="sigmoid")
    ])
    logging.info("Build RNN model successfully.")

    # Compole the RNN model
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info("Compile RNN model successfully.")

    return model


def lstm(input_dim, input_length):
    """Build LSTM model.

    Args:
        input_dim: int, the number of words in the vocabulary
        input_length: int, the length of input sequence.

    Returns:
        The built LSTM model
    """

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
        keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True),
        keras.layers.LSTM(units=128, dropout=0.2),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(rate=0.2),
        # Sigmoid is used for binary classification
        keras.layers.Dense(units=1, activation="sigmoid")
    ])
    logging.info("Build LSTM model successfully.")

    # Compole the LSTM model
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info("Compile LSTM model successfully.")

    return model


def pack_X(ham_texts, spam_texts, max_words, max_length):
    """Pack all type of texts data, i.e., X

    Args:
        ham_texts: list, store all ham mail texts of enron-spam dataset
        spam_texts: list, store all spam mail texts of enron-spam dataset
        max_words: int, the maximal number of words that are used
        max_length: int, the length of input sequence

    Returns:
        X_list: list, store all type of X in the form of array
        y: array, shape (n_samples), the labels of dataset
        vector_flag: list, store the flag that indicate the type of X in X_list
    """

    X_binary, y = get_binary_vector(ham_texts, spam_texts, max_words)
    X_tf, y = get_tf_vector(ham_texts, spam_texts, max_words)
    X_tfidf, y = get_tfidf_vector(ham_texts, spam_texts, max_words)
    X_vocab, y = get_vocabulary_vector(ham_texts, spam_texts, max_words, max_length)
    logging.info("Transforming original dataset into all type of X successfully")

    # Store all kind of X into a list
    X_list = [X_binary, X_tf, X_tfidf, X_vocab]
    vector_flags = [False, False, False, True]

    return X_list, y, vector_flags


def pack_models(vector_flag, max_words, max_length):
    """Pack all kind of models

    Args:
        vector_flag: bool, indicate the current type of X that is used
        max_words: int, the maximal number of words that are used
        max_length: int, the length of input sequence

    Returns:
        model_list: list, store the initialized model
        names: list, store the name of each model
        model_flags: list, store the string that indicates the type of model
    """

    # Vector_flag is False indicates the type of X that consists of binary, tf, tf-idf vectors
    if vector_flag == False:

        # Initialize all kind of models
        # gnb_model = GaussianNB()
        # svm_model = SVC(C=1.0)
        # mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2), activation="relu", solver="adam", batch_size=500, shuffle=True, verbose=True)
        # dnn_model = dnn(vector_length=max_words)
        # cnn_model = cnn(input_dim=max_words, input_length=max_words)
        rnn_model = rnn(input_dim=max_words, input_length=max_words)
        # lstm_model = lstm(input_dim=max_words, input_length=max_words)
        logging.info("Initialize all kind of model successfully")

    # Vector_flag is False indicates the type of X that consists of vocabulary vectors
    if vector_flag == True:

        # Initialize all kind of models
        gnb_model = GaussianNB()
        # svm_model = SVC(C=1.0)
        mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2), activation="relu", solver="adam", batch_size=500, shuffle=True, verbose=True)
        dnn_model = dnn(vector_length=max_length)
        cnn_model = cnn(input_dim=max_words, input_length=max_length)
        rnn_model = rnn(input_dim=max_words, input_length=max_length)
        lstm_model = lstm(input_dim=max_words, input_length=max_length)
        logging.info("Initialize all kind of model successfully")

    # Store all kind of initialized model into a list
    model_list = [gnb_model, mlp_model, dnn_model, cnn_model, rnn_model, lstm_model]

    # Store the type of all kind of model into a list
    model_flags = ["ML", "ML", "DL", "DL", "DL", "DL"]

    # Store the name of all kind of model into a list
    names = ["GaussianNB", "MLP", "DNN", "CNN", "RNN", "LSTM"]

    return model_list, model_flags, names


def train_model(model, X_train, y_train, epochs, batch_size, model_flag, name):
    """Train a model

    Args:
        model: the initialized model
        X_train: array, shape (n_samples, max_words/max_length), the texts data for training
        y_train: array, shape (n_samples), the labels data for training
        epochs: the number of epochs, only for DL model
        batch_size: the size of batch, only for DL model
        name: the name of the model

    Returns:
        the trained model
    """

    # For tradition ML model, fit without assigning epochs and batch_size
    if model_flag == "ML":

        logging.info("Begin training the {} model".format(name))
        model.fit(X_train, y_train)
        logging.info("Training {} model successfully".format(name))

    # For tradition DL model, fit with assigning epochs and batch_size
    if model_flag == "DL":

        logging.info("Begin training the {} model".format(name))
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        logging.info("Training {} model successfully".format(name))

    return model


def train_all_models(X_train, y_train, vector_flag, max_words, max_length, epochs, batch_size):
    """Train all models with a specific type of X

    Args:
        X_train: array, shape (n_samples, max_words/max_length), the texts data for training
        y_train: array, shape (n_samples), the labels data for training
        vector_flag: bool, indicate the type of X
        max_words: int, the maximal number of words that are used
        max_length: int, the length of input sequence
        epochs: the number of epochs, only for DL model
        batch_size: the size of batch, only for DL model

    Returns:
        trained_model_list: list, store all trained model
        names: list, store the name of each model
    """

    trained_model_list = []

    # Get the initialized models, model names and model flags according to the vector_flag
    model_list, model_flags, names = pack_models(vector_flag, max_words, max_length)

    for model, name, model_flag in zip(model_list, names, model_flags):

        # Train the model according to model_flag
        trained_model = train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size, model_flag=model_flag, name=name)

        # Add the trained model into the list
        trained_model_list.append(trained_model)

    return trained_model_list, model_flags, names


def test_model(trained_model, X_test, y_test, model_flag, name):
    """Test a model

    Args:
        trained_model: the trained model
        X_test: array, shape (n_samples, max_words/max_length), the texts data for testing
        y_test: array, shape (n_samples), the labels data for testing
        name: the name of the model

    Returns:
        acc_score: float, the test accuracy
    """

    if model_flag == "ML":

        logging.info("Begin testing the {} model".format(name))
        # Get the prediction of each sample in X_test
        y_test_predict = trained_model.predict(X_test)
        print(y_test_predict)
        logging.info("Tesing {} model successfully".format(name))

        # Get the test accuracy
        acc_score = accuracy_score(y_test, y_test_predict)

    if model_flag == "DL":

        logging.info("Begin testing the {} model".format(name))
        # Evaluate the trained model on test dataset
        score = trained_model.evaluate(X_test, y_test, verbose=1)
        logging.info("Tesing {} model successfully".format(name))

        acc_score = score[1]

    return acc_score


def test_all_models(trained_model_list, X_test, y_test, model_flags, names):
    """Test all models

    Args:
        trained_model_list: list, store all trained models
        X_test: array, shape (n_samples, max_words/max_length), the texts data for testing
        y_test: array, shape (n_samples), the labels data for testing
        names: list, store the name of each model

    Returns:
        acc_scores: list, the accuracy score of all models
    """

    # Store the accuracy score of all models
    accuracy_scores = []

    for trained_model, model_flag, name in zip(trained_model_list, model_flags, names):

        acc_score = test_model(trained_model, X_test, y_test, model_flag, name)
        accuracy_scores.append(acc_score)

    return accuracy_scores


if __name__ == "__main__":

    # Set maximal number of words used and the maximal length of sequence
    max_words = 5000
    max_length = 100

    # Set the number of epochs and size of batch
    epochs = 10
    batch_size = 500

    # The list for storing the accuracy score of all kind of models on each type of dataset
    acc_scores_all = []

    # Load ham texts and spam texts
    ham_texts, spam_texts = load_all_files()

    # Get X in the form of binary, tf, tfidf and vocabulary, and their corresponding flag
    X_list, y, vector_flags = pack_X(ham_texts, spam_texts, max_words, max_length)

    # Train and test each type of X on multiple models
    for X, vector_flag in zip(X_list, vector_flags):

        # Split X into train and test part
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

        # Training each type of model on training dataset
        trained_model_list, model_flags, names = train_all_models(X_train, y_train, vector_flag, max_words, max_length, epochs, batch_size)

        # Testing each type of model on testing dataset
        acc_scores = test_all_models(trained_model_list, X_test, y_test, model_flags, names)

        acc_scores_all.append(acc_scores)

    print("The accuracy scores result on test dataset is :", acc_scores_all)

    # Convert acc_scores_all into a dataframe
    acc_scores_all = pd.DataFrame(acc_scores_all, index=["Binary", "TF", "TF-IDF", "Vocabulary"], columns=names)

    # Plot the accuracy score result
    fig, ax = plt.subplots(figsize=(12, 8))
    acc_scores_all.plot(kind="bar", ax=ax)
    ax.set_xticklabels(rotation=90)
    plt.show()









