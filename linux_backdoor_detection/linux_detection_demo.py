#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This is a demo of linux backdoor detection, where ADFA-LD dataset is used

"""
import os
import glob
import logging
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)


def load_attack_samples():
    """Load all attack samples

    Returns:
        attack_texts: list of string, store all the attack samples
        attack_texts: list of int, store the symbol of attack sample
    """

    # Store texts and corresponding labels
    attack_texts = []
    attack_labels = []

    # Get global file path
    file_paths = glob.glob("data/Attack_Data_Master/*/*")

    for file_path in file_paths:

        with open(file_path) as file:

            logging.info("Successful open {}".format(file_path))
            lines = file.readlines()

        attack_texts.append(" ".join(lines))

        # 1 indicates attack sample
        attack_labels.append(1)

    return attack_texts, attack_labels


def load_normal_samples():
    """Load all normal samples

    Returns:
        normal_texts: list of string, store all normal samples
        normal_labels: list of int, store the symbol of normal sample
    """

    # Store texts and corresponding labels
    normal_texts = []
    normal_labels = []

    # Get global file path
    file_paths = glob.glob("data/Training_Data_Master/*")

    for file_path in file_paths:

        with open(file_path) as file:

            logging.info("Successfully open {}".format(file_path))
            lines = file.readlines()

        normal_texts.append(" ".join(lines))

        # 0 indicates normal sample
        normal_labels.append(0)

    return normal_texts, normal_labels


def load_validate_samples():
    """Load all validation samples

    Returns:
        validate_texts: list of string, store all validate samples
        validate_labels: list of int, store the symbol of validate sample
    """

    # Store texts and corresponding labels
    validate_texts = []
    validate_labels = []

    # Get global file path
    file_paths = glob.glob("data/Validation_Data_Master/*")

    for file_path in file_paths:

        with open(file_path) as file:

            logging.info("Successfully open {}".format(file_path))
            lines = file.readlines()

        validate_texts.append(" ".join(lines))

        # 0 indicates normal sample
        validate_labels.append(0)

    return validate_texts, validate_labels


def get_tfidf_dataset(max_words):
    """Get ti-idf vectorized dataset

    Args:
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the features for training
        y_train: array, shape (n_samples), the labels for training
        X_valid: array, shape (n_samples, max_words), the features for validating
        y_valid: array, shape (n_samples), the labels fro validating
    """

    # Load attack samples, normal and validate samples
    attack_texts, attack_labels = load_attack_samples()
    normal_texts, normal_labels = load_normal_samples()
    validate_texts, validate_labels = load_validate_samples()

    # Get the training texts and their labels
    texts_train = attack_texts + normal_texts
    labels_train = attack_labels + normal_labels

    # Initialize a tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer(decode_error="ignore", lowercase=True, stop_words="english",
                                         token_pattern=r"\b\d+\b", ngram_range=(3, 3), max_features=max_words, binary=False)

    # Transform all training texts as ti-idf vectors
    X_train = tfidf_vectorizer.fit_transform(texts_train).toarray()
    logging.info("Successfully transform training texts as tf-idf vectors")

    # Get y of training dataset
    y_train = np.array(labels_train)

    # Transform all validate texts as ti-idf vectors
    X_valid = tfidf_vectorizer.transform(validate_texts).toarray()
    logging.info("Successfully transform validate texts as tf-idf vectors")

    # Get y of validating dataset
    y_valid = np.array(validate_labels)

    # Shuffle dataset
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid


def train_model(model, X_train, y_train, name):
    """Train model

    Args:
        model: the initialized model
        X_train: array, shape (n_samples, max_words), the texts data for training
        y_train: array, shape (n_samples), the labels data for training
        name: the name of the model

    Returns:
        the trained model
    """

    logging.info("Begin training the {} model".format(name))
    model.fit(X_train, y_train)
    logging.info("Training {} model successfully".format(name))

    return model


def validate_model(trained_model, X_valid, y_valid, name):
    """Validate model

    Args:
        trained_model: the trained model
        X_valid: array, shape (n_samples, max_words), the texts data for validation
        y_valid: array, shape (n_samples), the labels data for validation
        name: the name of the model

    Returns:
        clf_report: the classification report
        cfu_matrix: the confusion matrix
    """

    logging.info("Begin validating the {} model".format(name))
    y_valid_predict = trained_model.predict(X_valid)
    logging.info("Validating {} model successfully".format(name))

    # Get classification report and confusion matrix
    clf_report = classification_report(y_valid, y_valid_predict)
    cfu_matrix = confusion_matrix(y_valid, y_valid_predict)

    return clf_report, cfu_matrix


if __name__ == "__main__":

    # Set the maximal number of words that are used
    max_words = 500

    # Get training dataset and validating dataset
    X_train, y_train, X_valid, y_valid = get_tfidf_dataset(max_words=max_words)

    # # Train and validate GaussianNB model on tf-idf dataset
    # gnb_model = GaussianNB()
    # trained_model = train_model(gnb_model, X_train, y_train, "GaussianNB")
    # clf_report, cfu_matrix = validate_model(trained_model, X_valid, y_valid, "GaussianNB")
    # print("Classification report: \n", clf_report)
    # print("Confusion matrix: \n", cfu_matrix)     # 2668 1704

    # # Train and validate MLP model on tf-idf dataset
    # mlp_model = MLPClassifier(hidden_layer_sizes=(10, 4))
    # trained_model = train_model(mlp_model, X_train, y_train, "MLP")
    # clf_report, cfu_matrix = validate_model(trained_model, X_valid, y_valid, "MLP")
    # print("Classification report: \n", clf_report)
    # print("Confusion matrix: \n", cfu_matrix)      # 3704 668

    # Train and validate Xgboost model on tf-idf dataset
    xgb_model = XGBClassifier(n_estimators=200, n_jobs=-1)
    trained_model = train_model(xgb_model, X_train, y_train, "Xgboost")
    clf_report, cfu_matrix = validate_model(trained_model, X_valid, y_valid, "Xgboost")
    print("Classification report: \n", clf_report)
    print("Confusion matrix: \n", cfu_matrix)        # 3737 635











