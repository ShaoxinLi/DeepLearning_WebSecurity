#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is demo of webshell detection, and only php files are used as data

"""
import os
import re
import logging
import warnings
import subprocess
import numpy as np
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)


def load_one_file(file_path):
    """Load one file

    Args:
        file_path: string, the path of file

    Returns:
        text: string, the content of file
    """

    text = ""

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

        logging.info("Load {} successfully".format(file_path))
        for line in file:

            # Remove \r and \n of each line
            line = line.strip("\r")
            line = line.strip("\n")
            text += line

    return text


def load_all_files(dir_path):
    """Load all php files or txt files under a directory

    Args:
        dir_path: string, the path of root directory

    Returns:
        texts: list of string, all content of php files or txt files
    """

    # Store text of each pgh file or txt file
    texts = []

    for root, dir_list, file_list in os.walk(dir_path):

        for file in file_list:

            # if this is a php file or txt file, load its content and save to texts
            if file.endswith(".php") or file.endswith(".txt"):

                # Get the path of current file
                file_path = os.path.join(root, file)
                logging.info("The current file path: {}".format(file_path))

                # Load the current file
                text = load_one_file(file_path)
                texts.append(text)

    return texts


def get_texts_labels(webshell_path, normal_path):
    """Get the texts of all webshell file and normal files, and their corresponding labels

    Args:
        webshell_path: string, the root path of all webshell files
        normal_path: string, the root path of all normal files

    Returns:
        texts: list of string, store all texts of webshell files and normal files
        labels: list of int, store all labels corresponding to texts
    """

    # Get all webshell texts
    texts_webshell = load_all_files(webshell_path)
    logging.info("Load all webshell texts successfully")

    # Get all normal texts
    texts_normal = load_all_files(normal_path)
    logging.info("Load all normal texts successfully")

    # Concat webshell samples and normal samples
    texts = texts_webshell + texts_normal

    # Count the number of webshell files and normal files
    n_webshell = len(texts_webshell)
    n_normal = len(texts_normal)

    # Construct labels, 0 indicates normal sample and 1 indicates webshell sample
    labels = [1] * n_webshell  + [0] * n_normal

    return texts, labels


def load_one_file_opcode(file_path, php_bin):
    """Load the opcodes of one php file

    Args:
        file_path: string, the path of php file

    Returns:
        text: string, store all opcodes of this php file
        n_tokens: int, the number of opcodes in this php file
    """

    # Get the command of executing vld
    cmd = php_bin + " -dvld.active=1 -dvld.execute=0 " + file_path

    # Get the output of executing command
    try:

        status, output = subprocess.getstatusoutput(cmd)
        logging.info("Successfully get the opcodes of {}".format(file_path))

        # Match the opcode
        tokens = re.findall(r'\s(\b[A-Z_]{2,}\b)\s', output)
        n_tokens = len(tokens)

        # Get the string of all opcode
        text = " ".join(tokens)
        logging.info("The number of opcodes is {}".format(n_tokens))

    except UnicodeDecodeError:

        text = ""
        n_tokens = 0
        logging.info("Fail to get the opcodes of {}".format(file_path))

    return text, n_tokens


def load_all_files_opcode(dir_path, min_opcode_count, php_bin):
    """Load the opcodes of all files under a directory

    Args:
        dir_path: string, the path of directory
        php_bin: string, the path of php_bin

    Returns:
        texts: store all texts of epcodes of php files
    """

    # Store text of each pgh file or txt file
    texts = []

    for root, dir_list, file_list in os.walk(dir_path):

        for file in file_list:

            # if this is a php file, load its content and save to texts
            if file.endswith(".php"):

                # Get the path of current file
                file_path = os.path.join(root, file)
                logging.info("The current file path: {}".format(file_path))

                # Load the current file
                text, n_tokens = load_one_file_opcode(file_path, php_bin)

                # If the number of opcode > min_opcode_count, then keep the current text
                if n_tokens > min_opcode_count:

                    texts.append(text)

                else:
                    logging.info("Load {} opcode failed".format(file_path))

    return texts


def get_texts_labels_opcode(webshell_path, normal_path, min_opcode_count, php_bin):
    """Get the texts of all webshell file and normal files, and their corresponding labels

    Args:
        webshell_path: string, the root path of all webshell files
        normal_path: string, the root path of all normal files
        min_opcode_count: int, the minimal number of opcodes in a php file
        php_bin: string, the path of php bin

    Returns:
        texts: list of string, store all texts of webshell files and normal files in the form of epcodes
        labels: list of int, store all labels corresponding to texts
    """

    # Get all webshell texts
    texts_webshell = load_all_files_opcode(webshell_path, min_opcode_count, php_bin)
    logging.info("Load all webshell texts successfully")

    # Get all normal texts
    texts_normal = load_all_files_opcode(normal_path, min_opcode_count, php_bin)
    logging.info("Load all normal texts successfully")

    # Concat webshell samples and normal samples
    texts = texts_webshell + texts_normal

    # Count the number of webshell files and normal files
    n_webshell = len(texts_webshell)
    n_normal = len(texts_normal)

    # Construct labels, 0 indicates normal sample and 1 indicates webshell sample
    labels = [1] * n_webshell  + [0] * n_normal

    return texts, labels


def get_tf_dataset(texts, labels, max_words):
    """Get the tf vectorized dataset

    Args:
        texts: list of string, store all texts of webshell files and normal files
        labels: list of int, store all labels corresponding to texts
        max_words: int, the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the features of training dataset
        X_test: array, shape (n_samples, max_words), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Split texts and labels into train and test part
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.3, shuffle=True)

    # Initialize a counter vectorizer
    tf_vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, token_pattern=r'\b\w+\b',
                                    max_features=max_words, ngram_range=(2, 2), binary=False)

    # Transform all train texts as tf vectors
    X_train = tf_vectorizer.fit_transform(texts_train).toarray()
    logging.info("Successfully transform training texts as tf vectors")

    # Transform all test texts as tf vectors
    X_test = tf_vectorizer.transform(texts_test).toarray()
    logging.info("Successfully transform testing texts as tf vectors")

    # Get y of training dataset
    y_train = np.array(labels_train)

    # Get y of testing dataset
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


def get_tfidf_dataset(texts, labels, max_words):
    """Get tf-idf vectorized dataset

    Args:
        texts: list of string, storing all texts
        labels: list of int, storing all labels corresponding to texts
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, output_dim), the features of training dataset
        X_test: array, shape (n_samples, output_dim), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Split train and test into train and test
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.3, shuffle=True)

    # Initialize a tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 3), decode_error="ignore", lowercase=True, stop_words="english", max_features=max_words, binary=False)

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



if __name__ == "__main__":

    # Set the paths of webshell files and normal files
    webshell_path = "data/webshell/PHP"
    normal_path = "data/normal/php"

    # Set the maximal number of words that are used and the dimensionality of output sequence
    max_words = 500
    output_dim = 100

    # Set the path of php bin
    php_bin = "/usr/bin/php7.2"

    # Set the number of minimal opcodes
    min_opcode_count = 2

    # # Train and test GaussianNB model on tf dataset by using php texts
    # texts, labels = get_texts_labels(webshell_path, normal_path)
    # X_train, X_test, y_train, y_test = get_tf_dataset(texts, labels, max_words)
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.81
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test GaussianNB model on tf dataset by using opcode texts
    # texts, labels = get_texts_labels_opcode(webshell_path, normal_path, min_opcode_count, php_bin)
    # X_train, X_test, y_train, y_test = get_tf_dataset(texts, labels, max_words)
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.80
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test MLP model on tf-idf dataset by using php texts
    # texts, labels = get_texts_labels(webshell_path, normal_path)
    # X_train, X_test, y_train, y_test = get_tfidf_dataset(texts, labels, max_words)
    # mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2), alpha=1e-5)
    # mlp_model.fit(X_train, y_train)
    # y_test_predicted = mlp_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.90
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # Train and test MLP model on tf dataset by using opcode texts
    texts, labels = get_texts_labels_opcode(webshell_path, normal_path, min_opcode_count, php_bin)
    X_train, X_test, y_train, y_test = get_tf_dataset(texts, labels, max_words)
    mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2), alpha=1e-5)
    mlp_model.fit(X_train, y_train)
    y_test_predicted = mlp_model.predict(X_test)
    print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.95
    print("Clf report: \n", classification_report(y_test, y_test_predicted))
    print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))
