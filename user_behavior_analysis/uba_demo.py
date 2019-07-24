#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This is a demo of user behavior analysis, where SEA dataset is used

"""
import os
import logging
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from hmmlearn import hmm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)


def load_data():
    """Load user's command lines

    Returns:
        texts: list of string, store all texts of user command
        labels: list of int, store all labels corresponding to texts
    """

    # Store all texts and corresponding labels
    texts = []
    labels = []

    for i in range(1, 51):

        # Set file path
        user_file_path = "data/masquerade-data/User{}".format(str(i))
        label_file_path = "data/labels.txt"

        # Load txt file
        user_sequences = np.loadtxt(user_file_path, dtype=str)
        logging.info("Successfully load {}".format(user_file_path))

        # Reshape command sequence to (150, 100)
        user_sequences = user_sequences.reshape((150, 100))

        # Transform sequence to a text
        for sequence in user_sequences:

            text = " ".join(sequence)
            texts.append(text)

        # Construct label
        user_labels = np.loadtxt(label_file_path, usecols=i-1, dtype=int).tolist()
        logging.info("Successfully load the {}th columns of {}".format(str(i), label_file_path))

        # Concat the first 50 zero
        user_labels = [0] * 50 + user_labels
        labels = labels + user_labels

    return texts, labels


def get_tf_dataset(texts, labels, max_words):
    """Get tf vectorized dataset

    Args:
        texts: list of string, store all texts of user command
        labels: list of int, store all labels corresponding to texts
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the features for training
        y_train: array, shape (n_samples), the labels for training
        X_test: array, shape (n_samples, max_words), the features for testing
        y_test: array, shape (n_samples), the labels for testing
    """

    # Split texts and labels into train and test
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.3, shuffle=True)

    # Initialize a counter vectorizer
    counter_vectorizer = CountVectorizer(decode_error="ignore", strip_accents="ascii", lowercase=True, stop_words="english",
                                         max_features=max_words, binary=False, ngram_range=(2, 4))

    # Transform all training texts as tf vectors
    X_train = counter_vectorizer.fit_transform(texts_train).toarray()
    logging.info("Successfully transform training texts as tf-idf vectors")

    # Transform all testing texts as tf vectors
    X_test = counter_vectorizer.transform(texts_test).toarray()
    logging.info("Transform testing text into tf vector successfully")

    # Get y of training dataset and testing dataset
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


def get_tfidf_dataset(texts, labels, max_words):
    """Get tf-idf vectorized dataset

    Args:
        texts: list of string, storing all texts
        labels: list of int, storing all labels corresponding to texts
        max_words: int, the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the features of training dataset
        X_test: array, shape (n_samples, max_words), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Split train and test into train and test
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.3, shuffle=True)

    # Initialize a tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer(decode_error="ignore", lowercase=True, stop_words="english", max_features=max_words, binary=False)

    # Transform training texts as tf-idf vectors
    X_train = tfidf_vectorizer.fit_transform(texts_train).toarray()
    logging.info("Transform training text into tf-idf vector successfully")

    # Transform testing texts as tf-idf vectors
    X_test = tfidf_vectorizer.transform(texts_test).toarray()
    logging.info("Transform training text into tf-idf vector successfully")

    # Get y of training dataset and testing dataset
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


def get_vocabulary_dataset(texts, labels, max_words, output_dim):
    """Get vocabulary vectorized dataset

    Args:
        texts: list of string, storing all texts
        labels: list of int, storing all labels corresponding to texts
        max_words: the maximal number of words that are used
        output_dim: the length of sequence

    Returns:
        X_train: array, shape (n_samples, max_words), the features of training dataset
        X_test: array, shape (n_samples, max_words), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Split texts and labels into train and test
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.3, shuffle=True)

    # Initialize a tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)

    # Fit on the training texts
    tokenizer.fit_on_texts(texts_train)

    # Transform training texts as padded vocabulary vectors
    X_train = tokenizer.texts_to_sequences(texts_train)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=output_dim)
    logging.info("Transform training text into vocabulary vector successfully")

    # Transform testing texts as padded vocabulary vectors
    X_test = tokenizer.texts_to_sequences(texts_test)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=output_dim)
    logging.info("Transform testing text into vocabulary vector successfully")

    # Get y of training dataset and testing dataset
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Set the maximal number of words that are used and the dimensionality of output sequence
    max_words = 500
    output_dim = 100

    # Set the number of epochs and size of batch
    epochs = 10
    batch_size = 128

    # Load texts and labels
    texts, labels = load_data()

    # # Train and test GaussianNB model on vocabulary dataset
    # X_train, X_test, y_train, y_test = get_vocabulary_dataset(texts, labels, max_words, output_dim)
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))     # 0.88
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test MLP model on vocabulary dataset
    # X_train, X_test, y_train, y_test = get_vocabulary_dataset(texts, labels, max_words, output_dim)
    # mlp_model = MLPClassifier(hidden_layer_sizes=(10, 4))
    # mlp_model.fit(X_train, y_train)
    # y_test_predicted = mlp_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))     # 0.96
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test HMM model on vocabulary dataset
    # X_train, X_test, y_train, y_test = get_vocabulary_dataset(texts, labels, max_words, output_dim)
    # hmm_model = hmm.MultinomialHMM(n_components=2, n_iter=100, algorithm="viterbi")
    # hmm_model.fit(X_train)
    #
    # print("Start probability: \n", hmm_model.startprob_)
    # print("Transition probability: \n", hmm_model.transmat_)
    # print("Emission probability: \n", hmm_model.emissionprob_)
    #
    # y_test_predicted = hmm_model.predict(X_test)        # multiple features is not supported in MultinomialHMM
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test Xgboost model on vocabulary dataset
    # X_train, X_test, y_train, y_test = get_vocabulary_dataset(texts, labels, max_words, output_dim)
    # xgb_model = XGBClassifier(n_estimators=100, n_jobs=-1)
    # xgb_model.fit(X_train, y_train)
    # y_test_predicted = xgb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))     # 0.97
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))















