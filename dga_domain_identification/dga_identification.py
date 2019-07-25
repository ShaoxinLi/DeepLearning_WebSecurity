#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a demo of dga domain identification, where alexa data is used as white samples and 360 netlab data is used as
black samples

"""
import os
import re
import logging
import warnings
import numpy as np
import pandas as pd
from tensorflow import keras

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)


def load_alex_data():
    """Get all domains of alexa data

    Returns:
        domains: array, shape (n_sampels, ), store all domains
    """

    # Set the path of file
    file_path = "data/top-1m.csv"

    # Load csv data
    data = pd.read_csv(file_path, header=None, index_col=0, names=["index", "domain"])
    logging.info("Load {} successfully".format(file_path))

    # Get all domains
    domains = data.domain.values

    return domains[:10000]


def load_netlab_data():
    """Get all domains of 360 netlab data

    Returns:
        domains: array, shape (n_sampels, ), store all domains
    """

    # Set the path of file
    file_path = "data/dga.txt"

    # Load txt data
    data = pd.read_table(file_path, sep="\t", header=None, skiprows=18, names=["index", "domain", "data1", "data2"],
                         index_col="index")
    logging.info("Load {} successfully".format(file_path))

    # Get all domains
    domains = data.domain.values

    return domains[:10000]


def get_tf_dataset(max_words):
    """Get tf vectorized dataset

    Args:
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the features of training dataset
        X_test: array, shape (n_samples, max_words), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Get alex domains and netlab domains
    alex_domains = load_alex_data()
    netlab_domains = load_netlab_data()

    # Concat all domains
    domains = np.concatenate([alex_domains, netlab_domains])

    # Construct labels corresponding tp domains, 0 indicates white sample and 1 indicates black sample
    labels = [0] * len(alex_domains) + [1] * len(netlab_domains)

    # Split domains and labels into train and test part
    domains_train, domains_test, labels_train, labels_test = train_test_split(domains, labels, test_size=0.3, shuffle=True)

    # Initialize a tf vectorizer
    tf_vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, stop_words='english', token_pattern='\w',
                                    ngram_range=(2, 2), max_features=max_words, binary=False)


    # Transform training domains to vectors
    X_train = tf_vectorizer.fit_transform(domains_train).toarray()
    logging.info("Transform training domains into tf vector successfully")

    # Transform testing domains to vectors
    X_test = tf_vectorizer.transform(domains_test).toarray()
    logging.info("Transform testing domains into tf vector successfully")

    # Get y of training dataset and testing dataset
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


def get_tfidf_dataset(max_words):
    """Get the tf-idf vectorized dataset

    Args:
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the features of training dataset
        X_test: array, shape (n_samples, max_words), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Get alex domains and netlab domains
    alex_domains = load_alex_data()
    netlab_domains = load_netlab_data()

    # Concat all domains
    domains = np.concatenate([alex_domains, netlab_domains])

    # Construct labels corresponding tp domains, 0 indicates white sample and 1 indicates black sample
    labels = [0] * len(alex_domains) + [1] * len(netlab_domains)

    # Split domains and labels into train and test part
    domains_train, domains_test, labels_train, labels_test = train_test_split(domains, labels, test_size=0.3, shuffle=True)

    # Initialize a tf vectorizer
    tfidf_vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, stop_words='english', token_pattern='\w',
                                    ngram_range=(2, 2), max_features=max_words, binary=False)

    # Transform training domains to vectors
    X_train = tfidf_vectorizer.fit_transform(domains_train).toarray()
    logging.info("Transform training domains into tf-idf vector successfully")

    # Transform testing domains to vectors
    X_test = tfidf_vectorizer.transform(domains_train).toarray()
    logging.info("Transform testing domains into tf-idf vector successfully")

    # Get y of training dataset and testing dataset
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


def get_charseq_dataset(max_length):
    """Get the dataset that consists of char sequence of each domain

    Args:
        max_length: the maximal length of each sequence

    Returns:
        X_train: array, shape (n_samples, max_length), the features of training dataset
        X_test: array, shape (n_samples, max_length), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Get alex domains and netlab domains
    alex_domains = load_alex_data()
    netlab_domains = load_netlab_data()

    # Concat all domains
    domains = np.concatenate([alex_domains, netlab_domains])

    # Construct labels corresponding tp domains, 0 indicates white sample and 1 indicates black sample
    labels = [0] * len(alex_domains) + [1] * len(netlab_domains)

    # Store all sequences corresponding to domains
    domain_sequences = []

    # Transform domains as char sequences
    for domain in domains:

        domain_sequence = []

        for i in range(len(domain)):

            domain_sequence.append(ord(domain[i]))

        domain_sequences.append(domain_sequence)

    # Split domains and labels into train and test part
    domains_train, domains_test, labels_train, labels_test = train_test_split(domain_sequences, labels, test_size=0.3,
                                                                              shuffle=True)

    # Pad all sequences of training dataset
    X_train = keras.preprocessing.sequence.pad_sequences(domains_train, maxlen=max_length)
    logging.info("Transform training domains into vectors successfully")

    # Pad all sequences of testing dataset
    X_test = keras.preprocessing.sequence.pad_sequences(domains_test, maxlen=max_length)
    logging.info("Transform testing domains into vectors successfully")

    # Get y of training dataset and testing dataset
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    return X_train, X_test, y_train, y_test


def jarccard_index(domian_1, domina_2):
    """Compute the jarccard index between two domains

    Args:
        domian_1: string
        domina_2: string

    Returns:
        float, the computed jarccard index
    """

    # Construct a set (2-gram) corresponding to domian_1
    x = set(' ' + domian_1[0])

    for i in range(len(x)-1):

        x.add(domian_1[i] + domian_1[i+1])

    x.add(domian_1[len(domian_1)-1] + ' ')

    # Construct a set (2-gram) corresponding to domian_2
    y = set(' ' + domina_2[0])

    for i in range(len(y)-1):

        y.add(domina_2[i] + domina_2[i+1])

    y.add(domina_2[len(domina_2)-1] + ' ')

    return len(x&y) / len(x|y)


def get_stat_char_dataset():
    """Get a dataset with statistical features

    Returns:
        X_train: array, shape (n_samples, 3), the features of training dataset
        X_test: array, shape (n_samples, 3), the features of testing dataset
        y_train, array, shape (n_samples), the labels of training dataset
        y_test, array, shape (n_samples), the labels of testing dataset
    """

    # Get alex domains and netlab domains
    alex_domains = load_alex_data()
    netlab_domains = load_netlab_data()

    # Concat all domains
    domains = np.concatenate([alex_domains, netlab_domains])

    # Construct labels corresponding tp domains, 0 indicates white sample and 1 indicates black sample
    labels = [0] * len(alex_domains) + [1] * len(netlab_domains)

    # Get statistical features of each domain
    stat_features = []

    for domain in domains:

        # Compute the ratio of the length of vowel chars to the length of entire domain
        vowel_count = len(re.findall(r'[aeiou]', domain.lower())) / len(domain)

        # Compute the ratio of the length of unique chars to the length of entire domain
        unique_char_count = len(set(domain)) / len(domain)

        # Compute the jarccard index of the current domain
        j_index = -1

        for i in domains:

            j_index += jarccard_index(domain, i)

        j_index = j_index/(len(domains)-1)

        stat_features.append([vowel_count, unique_char_count, j_index])

    # Transform list as array
    stat_features = np.array(stat_features)

    # Split domains and labels into train and test part
    X_train, X_test, y_train, y_test = train_test_split(stat_features, labels, test_size=0.3,
                                                                              shuffle=True)

    return X_train, X_test, y_train, y_test


def dnn(input_dim):
    """Build DNN model

    Args:
        input_dim: int, the dimensionality of input

    Returns:
        the built DNN model
    """

    # DNN model structure
    model = keras.models.Sequential([
        keras.layers.Dense(units=512, activation='relu', input_shape=(input_dim,)),
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


def rnn(input_dim, input_length):
    """Build RNN model.

    Args:
        input_dim: the number of words that are used in the embedding layer
        input_length: the length of input sequence

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

    # Compile the RNN model
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info("Compile RNN model successfully.")

    return model


if __name__ == "__main__":

    # Set the maximal number of words that are used
    max_words = 500
    max_length = 100

    # # Train and test GaussianNB model on tf dataset
    # X_train, X_test, y_train, y_test = get_tf_dataset(max_words=max_words)
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))     # 0.91
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test GaussianNB model on statistical features dataset
    # X_train, X_test, y_train, y_test = get_stat_char_dataset()
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.78
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test Xgboost model on tf dataset
    # X_train, X_test, y_train, y_test = get_tf_dataset(max_words=max_words)
    # xgb_model = XGBClassifier(n_estimators=200, n_jobs=-1)
    # xgb_model.fit(X_train, y_train)
    # y_test_predicted = xgb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.87
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test Xgboost model on statistical features dataset
    # X_train, X_test, y_train, y_test = get_stat_char_dataset()
    # xgb_model = XGBClassifier(n_estimators=200, n_jobs=-1)
    # xgb_model.fit(X_train, y_train)
    # y_test_predicted = xgb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.80
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test DNN model on tf dataset
    # X_train, X_test, y_train, y_test = get_tf_dataset(max_words=max_words)
    # dnn_model = dnn(input_dim=max_words)
    # dnn_model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)
    # score = dnn_model.evaluate(X_test, y_test)
    # print("Accuracy score: \n", score[1])       # 0.94

    # # Train and test DNN model on statistical features dataset
    # X_train, X_test, y_train, y_test = get_stat_char_dataset()
    # dnn_model = dnn(input_dim=2)
    # dnn_model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)
    # score = dnn_model.evaluate(X_test, y_test)
    # print("Accuracy score: \n", score[1])       # 0.77

    # # Train and test RNN model on charseq dataset
    # X_train, X_test, y_train, y_test = get_charseq_dataset(max_length=max_length)
    # rnn_model = rnn(input_dim=max_words, input_length=max_length)
    # rnn_model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1)
    # score = rnn_model.evaluate(X_test, y_test)
    # print("Accuracy score: \n", score[1])       # 0.96
