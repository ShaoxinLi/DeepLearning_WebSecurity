#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a demo of credit card detection, where a dataset on kaggle is used

"""
import logging
import warnings
import numpy as np
import pandas as pd
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')


def load_data():
    """Load csv data file

    Returns:
        data: Dataframe, the credit card dataset
    """

    # Set the path of data file
    data_path = "data/creditcard.csv"

    # Load data file
    data = pd.read_csv(data_path)
    logging.info("Load {} successfully".format(data_path))

    return data


def normalizating(data):
    """Normalize the features of dataset

    Args:
        data: Dataframe, the credit card dataset

    Returns:
        data: Dataframe, the normalized dataset
    """

    # Initializer a scaler
    scaler = StandardScaler()

    # Normalize feature Amount as normAmount
    data['normAmount'] = scaler.fit_transform(data[['Amount']])
    logging.info("Normalize feature Amount successfully")

    # Drop features Time and Amount
    data.drop(['Time', 'Amount'], axis=1, inplace=True)

    return data


def undersampling(data):
    """Get the training dataset and testing dataset by using undersampling

    Args:
        data: Dataframe, the normalized dataset

    Returns:
        X_train: array, shape (n_samples, n_features), the features of training dataset
        X_test: array, shape (n_samples, n_features), the features of testing dataset
        y_train: array, shape (n_samples, ), the labels of training dataset
        y_test: array, shape (n_samples, ), the labels of testing dataset
    """

    # Get the number of black samples
    n_fraud = len(data[data.Class==1])

    # Get the indices of black samples
    fraud_indices = np.array(data[data.Class==1].index)

    # Get the indices of white samples
    normal_indices = np.array(data[data.Class==0].index)

    # Get the indices of the sampled normal samples
    random_normal_indices = np.random.choice(normal_indices, size=n_fraud, replace=False)

    # Get the all indices
    X_indices = np.concatenate([fraud_indices, random_normal_indices])

    # Get all features X
    data.drop(['Class'], axis=1, inplace=True)
    X = data.iloc[X_indices, :].values

    # Get all labels y
    y = [1] * n_fraud + [0] * n_fraud
    y = np.array(y)

    # Split X, y into train an test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    return X_train, X_test, y_train, y_test


def oversampling(data):
    """Get the training dataset and testing dataset by using oversampling

    Args:
        data: Dataframe, the normalized dataset

    Returns:
        X_train: array, shape (n_samples, n_features), the features of training dataset
        X_test: array, shape (n_samples, n_features), the features of testing dataset
        y_train: array, shape (n_samples, ), the labels of training dataset
        y_test: array, shape (n_samples, ), the labels of testing dataset
    """

    # Get all labels y
    y = data['Class'].values

    # Get all features X
    data.drop(['Class'], axis=1, inplace=True)
    X = data.values

    # Split X, y into train and test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    # Oversampling training dataset by using smote algorithm
    smote = SMOTE()
    X_train, y_train = smote.fit_sample(X_train, y_train)
    logging.info("Oversampling successfully")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Load credit card dataset
    data = load_data()

    # Normalize some features
    normalized_data = normalizating(data)

    # # Train and test GaussianNB on undersampling dataset
    # X_train, X_test, y_train, y_test = undersampling(normalized_data)
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.93
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))    # recall: 0.90
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test GaussianNB on oversampling dataset
    # X_train, X_test, y_train, y_test = oversampling(normalized_data)
    # gnb_model = GaussianNB()
    # gnb_model.fit(X_train, y_train)
    # y_test_predicted = gnb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.97
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))    # recall: 0.82
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test Xgboost on undersampling dataset
    # X_train, X_test, y_train, y_test = undersampling(normalized_data)
    # xgb_model = XGBClassifier(n_estimators=100, n_jobs=-1)
    # xgb_model.fit(X_train, y_train)
    # y_test_predicted = xgb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.93
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))    # recall: 0.93
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test Xgboost on oversampling dataset
    # X_train, X_test, y_train, y_test = oversampling(normalized_data)
    # xgb_model = XGBClassifier(n_estimators=100, n_jobs=-1)
    # xgb_model.fit(X_train, y_train)
    # y_test_predicted = xgb_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.98
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))    # recall: 0.89
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # # Train and test MLP on undersampling dataset
    # X_train, X_test, y_train, y_test = undersampling(normalized_data)
    # mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2))
    # mlp_model.fit(X_train, y_train)
    # y_test_predicted = mlp_model.predict(X_test)
    # print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.93
    # print("Clf report: \n", classification_report(y_test, y_test_predicted))    # recall: 0.93
    # print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))

    # Train and test MLP on oversampling dataset
    X_train, X_test, y_train, y_test = oversampling(normalized_data)
    mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2))
    mlp_model.fit(X_train, y_train)
    y_test_predicted = mlp_model.predict(X_test)
    print("Accuracy score: \n", accuracy_score(y_test, y_test_predicted))       # 0.99
    print("Clf report: \n", classification_report(y_test, y_test_predicted))    # recall: 0.82
    print("Confusion matrix: \n", confusion_matrix(y_test, y_test_predicted))






