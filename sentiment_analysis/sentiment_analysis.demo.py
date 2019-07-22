#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a demo of sentiment analysis, where imdb dataset is used.

"""
import os
import logging
import multiprocessing
import numpy as np
import keras
import tensorflow as tf
from collections import namedtuple

from gensim.models import Word2Vec
from gensim.models import Doc2Vec

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

config = tf.ConfigProto(device_count = {"CPU": 4})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
logging.getLogger().setLevel(logging.INFO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_one_file(file_path):
    """Load single file.

    Args:
        file_path: string, the path of file

    Returns:
        text: string, the cotent of this file
    """

    with open(file_path) as file:

        logging.info("Load mail file {} successfully".format(file_path))
        text = ""

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
        pos_texts_train: list, store all positive texts of training dataset
        neg_texts_train: list, store all negative texts of training dataset
        pos_texts_test: list, store all positive texts of testing dataset
        neg_texts_test: list, store all negative texts of testing dataset
    """

    # The path of dataset
    pos_texts_train_path = "data/train/pos"
    neg_texts_train_path = "data/train/neg"
    pos_texts_test_path = "data/test/pos"
    neg_texts_test_path = "data/test/neg"

    # Load all files
    pos_texts_train = load_files_from_dir(pos_texts_train_path)
    logging.info("Load all positive texts of training dataset successfully")

    neg_texts_train = load_files_from_dir(neg_texts_train_path)
    logging.info("Load all negative texts of training dataset successfully")

    pos_texts_test = load_files_from_dir(pos_texts_test_path)
    logging.info("Load all positive texts of testing dataset successfully")

    neg_texts_test = load_files_from_dir(neg_texts_test_path)
    logging.info("Load all negative texts of testing dataset successfully")

    return pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test


def get_tf_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, max_words):
    """Get the tf vector of each text

    Args:
        pos_texts_train: list, store all positive texts of training dataset
        neg_texts_train: list, store all negative texts of training dataset
        pos_texts_test: list, store all positive texts of testing dataset
        neg_texts_test: list, store all negative texts of testing dataset
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the tf vectorized text data for training
        X_test: array, shape (n_samples, max_words), the tf vectorized text data for testing
        y_train: array, shape (n_smaples), the labels of training data
        y_test: array, shape (n_smaples), the labels of testing data
    """

    # Get all training texts
    texts_train = pos_texts_train + neg_texts_train

    # Get all testing texts
    texts_test = pos_texts_test + neg_texts_test

    # Initialize a counter vectorizer
    counter_vectorizer = CountVectorizer(decode_error="ignore", lowercase=True, stop_words="english", max_features=max_words, binary=False)

    # Transform training texts as tf vectors
    X_train = counter_vectorizer.fit_transform(texts_train).toarray()
    logging.info("Transform training text into tf vector successfully")

    # Transform testing texts as tf vectors
    X_test = counter_vectorizer.transform(texts_test).toarray()
    logging.info("Transform training text into tf vector successfully")

    # Construct train labels array, 0 indicates positive texts, 1 indicates negative texts
    y_train = [0] * len(pos_texts_train) + [1] * len(neg_texts_train)
    y_train = np.asarray(y_train)

    # Construct test labels array
    y_test = [0] * len(pos_texts_test) + [1] * len(neg_texts_test)
    y_test = np.asarray(y_test)

    # Shuffle both train data and test data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test


def get_tfidf_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, max_words):
    """Get the tf-idf vector of each text

    Args:
        pos_texts_train: list, store all positive texts of training dataset
        neg_texts_train: list, store all negative texts of training dataset
        pos_texts_test: list, store all positive texts of testing dataset
        neg_texts_test: list, store all negative texts of testing dataset
        max_words: the maximal number of words that are used

    Returns:
        X_train: array, shape (n_samples, max_words), the tf vectorized text data for training
        X_test: array, shape (n_samples, max_words), the tf vectorized text data for testing
        y_train: array, shape (n_smaples), the labels of training data
        y_test: array, shape (n_smaples), the labels of testing data
    """

    # Get all training texts
    texts_train = pos_texts_train + neg_texts_train

    # Get all testing texts
    texts_test = pos_texts_test + neg_texts_test

    # Initialize a tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer(decode_error="ignore", lowercase=True, stop_words="english", max_features=max_words, binary=False)

    # Transform training texts as tf-idf vectors
    X_train = tfidf_vectorizer.fit_transform(texts_train).toarray()
    logging.info("Transform training text into tf vector successfully")

    # Transform testing texts as tf-idf vectors
    X_test = tfidf_vectorizer.transform(texts_test).toarray()
    logging.info("Transform training text into tf vector successfully")

    # Construct train labels array, 0 indicates positive texts, 1 indicates negative texts
    y_train = [0] * len(pos_texts_train) + [1] * len(neg_texts_train)
    y_train = np.asarray(y_train)

    # Construct test labels array
    y_test = [0] * len(pos_texts_test) + [1] * len(neg_texts_test)
    y_test = np.asarray(y_test)

    # Shuffle both train data and test data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test


def get_vocabulary_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, max_words, output_dim):
    """Get the vocabulary vector of each text

     Args:
        pos_texts_train: list, store all positive texts of training dataset
        neg_texts_train: list, store all negative texts of training dataset
        pos_texts_test: list, store all positive texts of testing dataset
        neg_texts_test: list, store all negative texts of testing dataset
        max_words: the maximal number of words that are used
        output_dim: the length of sequence

     Returns:
        X_train: array, shape (n_samples, max_words), the tf vectorized text data for training
        X_test: array, shape (n_samples, max_words), the tf vectorized text data for testing
        y_train: array, shape (n_smaples), the labels of training data
        y_test: array, shape (n_smaples), the labels of testing data
     """

    # Get all training texts
    texts_train = pos_texts_train + neg_texts_train

    # Get all testing texts
    texts_test = pos_texts_test + neg_texts_test

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

    # Construct train labels array, 0 indicates positive texts, 1 indicates negative texts
    y_train = [0] * len(pos_texts_train) + [1] * len(neg_texts_train)
    y_train = np.asarray(y_train)

    # Construct test labels array
    y_test = [0] * len(pos_texts_test) + [1] * len(neg_texts_test)
    y_test = np.asarray(y_test)

    # Shuffle both train data and test data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test


def clean_texts(texts):
    """Clean texts

    Args:
        texts: list of string, store the texts, and its element is string

    Returns:
        texts: list of list of string, store the texts, and its element is a list that stores the words of text
    """

    punctuations = """.,?!:;(){}[]"""

    # Remove \n and lower text
    texts = [text.lower().replace('\n', '') for text in texts]

    # Remove <br >
    texts = [text.replace('<br />', ' ') for text in texts]

    # Treat punctuations as individual words
    for punctuation in punctuations:

        texts = [text.replace(punctuation, '{}'.format(punctuation)) for text in texts]

    texts = [text.split() for text in texts]

    return texts


def get_word2vec_features(word2vec_model, texts, output_dim):
    """Get the X in the form of word2vec

    Args:
        word2vec_model: the trained word2vec model
        texts: list of list of string, store the texts, and its element is a list that stores the words of text
        output_dim: int, the length of output dimensionality

    Returns:
        X: array, shape (n_samples, output_dim), vectorized texts
    """

    X = []

    for text in texts:

        vectors_of_one_text = np.zeros(output_dim)
        n_words = 0

        for word in text:

            try:

                # Get the vector of this word, return a numpy array, shape (output_dim,)
                vectorized_word = word2vec_model[word]

                # Compute the sum of vectors and add one to n_words
                vectors_of_one_text += vectorized_word
                n_words += 1

            except KeyError:

                continue

        # Get the average of vectors of this text, which is appended to X
        X.append(vectors_of_one_text/n_words)

    X = np.array(X, dtype="float")

    return X


def get_word2vec_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, output_dim):
    """Get the word2vec vector of each text

    Args:
        pos_texts_train: list, store all positive texts of training dataset
        neg_texts_train: list, store all negative texts of training dataset
        pos_texts_test: list, store all positive texts of testing dataset
        neg_texts_test: list, store all negative texts of testing dataset
        output_dim: int, the length of output dimensionality

    Returns:
        X_train: array, shape (n_samples, output_dim), the word2vec vectorized text data for training
        X_test: array, shape (n_samples, output_dim), the word2vec vectorized text data for testing
        y_train: array, shape (n_smaples), the labels of training data
        y_test: array, shape (n_smaples), the labels of testing data
    """

    # Clean texts
    pos_texts_train = clean_texts(pos_texts_train)
    neg_texts_train = clean_texts(neg_texts_train)
    pos_texts_test = clean_texts(pos_texts_test)
    neg_texts_test = clean_texts(neg_texts_test)

    # Get all training texts
    texts_train = pos_texts_train + neg_texts_train
    logging.info("The number of samples for training is {}".format(len(texts_train)))

    # Get all testing texts
    texts_test = pos_texts_test + neg_texts_test
    logging.info("The number of samples for testing is {}".format(len(texts_train)))

    # Get the number of CPU core
    cores= multiprocessing.cpu_count()

    # Initialize a Word2Vec model
    word2vec_model = Word2Vec(size=output_dim, window=10, min_count=10, iter=10, workers=cores)

    # Build vocabulary based on train text
    word2vec_model.build_vocab(texts_train)

    # Train the Word2Vec model on train texts and save
    word2vec_model.train(texts_train, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
    word2vec_model.save("word2vec.bin")

    # Transform training texts as word2vectors
    X_train = get_word2vec_features(word2vec_model, texts_train, output_dim)
    logging.info("Transform training text into word2vector successfully")

    # Transform testing texts as word2vectors
    X_test = get_word2vec_features(word2vec_model, texts_test, output_dim)
    logging.info("Transform testing text into word2vector successfully")

    # Construct train labels array, 0 indicates positive texts, 1 indicates negative texts
    y_train = [0] * len(pos_texts_train) + [1] * len(neg_texts_train)
    y_train = np.asarray(y_train)

    # Construct test labels array
    y_test = [0] * len(pos_texts_test) + [1] * len(neg_texts_test)
    y_test = np.asarray(y_test)

    # Shuffle both train data and test data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test


def labelize_texts(texts, text_type):
    """Labelize texts

    Args:
        texts: list of list of string, the texts for labeling
        label_type:

    Returns:
        labelized_texts: list of SentimentDocument
    """

    # Defined a nametuple
    SentimentDocument = namedtuple('SentimentDocument', ['words', 'tags'])

    # A list for storing all labelized texts
    labelized_texts = []

    for index, text in enumerate(texts):

        text_id_type = '{}_{}'.format(text_type, index)

        labelized_texts.append(SentimentDocument(words=text, tags=[text_id_type]))

    return labelized_texts


def get_doc2vec_feaures(doc2vec_model, texts):
    """Get the X in the form of doc2vec

    Args:
        doc2vec_model: the trained doc2vec model
        texts: list of list of string, the texts for vectoring
    Returns:
        array, shape (n_samples, output_dim)
    """

    vectors = [doc2vec_model.infer_vector(text) for text in texts]

    return np.array(vectors, dtype='float')


def get_doc2vec_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, output_dim):
    """Get the word2vec vector of each text

    Args:
        pos_texts_train: list, store all positive texts of training dataset
        neg_texts_train: list, store all negative texts of training dataset
        pos_texts_test: list, store all positive texts of testing dataset
        neg_texts_test: list, store all negative texts of testing dataset
        output_dim: int, the length of output dimensionality

    Returns:
        X_train: array, shape (n_samples, output_dim), the doc2vec vectorized text data for training
        X_test: array, shape (n_samples, output_dim), the doc2vec vectorized text data for testing
        y_train: array, shape (n_smaples), the labels of training data
        y_test: array, shape (n_smaples), the labels of testing data
    """

    pos_texts_train = clean_texts(pos_texts_train)
    neg_texts_train = clean_texts(neg_texts_train)
    pos_texts_test = clean_texts(pos_texts_test)
    neg_texts_test = clean_texts(neg_texts_test)

    # Get all training texts
    texts_train = pos_texts_train + neg_texts_train
    logging.info("The number of samples for training is {}".format(len(texts_train)))

    # Get all testing texts
    texts_test = pos_texts_test + neg_texts_test
    logging.info("The number of samples for testing is {}".format(len(texts_train)))

    # Labelize train texts
    texts_train_labelized = labelize_texts(texts_train, 'Train')

    # Get the number of CPU core
    cores= multiprocessing.cpu_count()

    # Initialize a Doc2Vec model
    doc2vec_model = Doc2Vec(size=output_dim, window=10, min_count=2, iter=60, workers=cores, hs=0, negative=5)

    # Build vocabulary based on train text
    doc2vec_model.build_vocab(texts_train_labelized)

    # Train the Doc2Vec model on train texts and save
    doc2vec_model.train(texts_train_labelized, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)
    doc2vec_model.save("doc2vec.bin")

    # Transform training texts as doc2vectors
    X_train = get_doc2vec_feaures(doc2vec_model, texts_train)
    logging.info("Transform training text into doc2vector successfully")

    # Transform testing texts as doc2vectors
    X_test = get_doc2vec_feaures(doc2vec_model,texts_test)
    logging.info("Transform testing text into doc2vector successfully")

    # Construct train labels array, 0 indicates positive texts, 1 indicates negative texts
    y_train = [0] * len(pos_texts_train) + [1] * len(neg_texts_train)
    y_train = np.asarray(y_train)

    # Construct test labels array
    y_test = [0] * len(pos_texts_test) + [1] * len(neg_texts_test)
    y_test = np.asarray(y_test)

    # Shuffle both train data and test data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test


def dnn(input_dim):
    """Build DNN model

    Args:
        input_dim: int, the length of input vector

    Returns:
        the built dnn model
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
        # There must be a flatten layer because the output of the MaxPooling1D layer is still a tensor
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

    # Compile the RNN model
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


def train_model(model, X_train, y_train, epochs, batch_size, model_flag, name):
    """Train a model

    Args:
        model: the initialized model
        X_train: array, shape (n_samples, max_words/max_length), the texts data for training
        y_train: array, shape (n_samples), the labels data for training
        epochs: the number of epochs, only for DL model
        batch_size: the size of batch, only for DL model
        model_flag: infer the type of model for training
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


def test_model(trained_model, X_test, y_test, model_flag, name):
    """Test a model

    Args:
        trained_model: the trained model
        X_test: array, shape (n_samples, max_words/max_length), the texts data for testing
        y_test: array, shape (n_samples), the labels data for testing
        model_flag: infer the type of model for training
        name: the name of the model

    Returns:
        acc_score: float, the test accuracy
    """

    if model_flag == "ML":

        logging.info("Begin testing the {} model".format(name))
        # Get the prediction of each sample in X_test
        y_test_predict = trained_model.predict(X_test)
        logging.info("Tesing {} model successfully".format(name))

        # Get the test accuracy
        acc_score = accuracy_score(y_test, y_test_predict)

    if model_flag == "DL":

        logging.info("Begin testing the {} model".format(name))
        # Evaluate the trained model on test dataset
        score = trained_model.evaluate(X_test, y_test, verbose=1)
        logging.info("Tesing {} model successfully".format(name))

        # Get the test accuracy
        acc_score = score[1]

    return acc_score


if __name__ == "__main__":

    # Set maximal number of words used and the length of sequence
    max_words = 500
    output_dim = 100

    # Set the number of epochs and size of batch
    epochs = 10
    batch_size = 128

    # Load all positive and negative texts for training and testing
    pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test = load_all_files()

    # # Train and test GaussianNB model on tf dataset
    # X_train, X_test, y_train, y_test = get_tf_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, max_words=max_words)
    # gnb_model = GaussianNB()
    # trained_model = train_model(gnb_model, X_train, y_train, epochs, batch_size, "ML", "GaussianNB")
    # accuracy_score = test_model(trained_model, X_test, y_test, "ML", "GaussianNB")
    # print(accuracy_score)

    # # Train and test SVM model on tf-idf dataset
    # X_train, X_test, y_train, y_test = get_tfidf_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, max_words=max_words)
    # svm_model = SVC()
    # trained_model = train_model(svm_model, X_train, y_train, epochs, batch_size, "ML", "SVM")
    # accuracy_score = test_model(trained_model, X_test, y_test, "ML", "SVM")
    # print(accuracy_score)

    # # Train and test CNN model on vocabulary dataset
    # X_train, X_test, y_train, y_test = get_vocabulary_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, max_words=max_words, output_dim=output_dim)
    # cnn_model = cnn(input_dim=max_words, input_length=output_dim)
    # trained_model = train_model(cnn_model, X_train, y_train, epochs, batch_size, "DL", "CNN")
    # accuracy_score = test_model(trained_model, X_test, y_test, "DL", "CNN")
    # print(accuracy_score)

    # # Train and test MLP model on word2vec dataset
    # X_train, X_test, y_train, y_test = get_word2vec_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, output_dim=output_dim)
    # mlp_model = MLPClassifier(hidden_layer_sizes=(5, 2), activation="relu", solver="adam", batch_size=batch_size, shuffle=True, verbose=True)
    # trained_model = train_model(mlp_model, X_train, y_train, epochs, batch_size, "ML", "MLP")
    # accuracy_score = test_model(trained_model, X_test, y_test, "ML", "MLP")
    # print(accuracy_score)

    # Train and test DNN model on doc2vec dataset
    X_train, X_test, y_train, y_test = get_doc2vec_dataset(pos_texts_train, neg_texts_train, pos_texts_test, neg_texts_test, output_dim=output_dim)
    dnn_model = dnn(input_dim=output_dim)
    trained_model = train_model(dnn_model, X_train, y_train, epochs, batch_size, "DL", "DNN")
    accuracy_score = test_model(trained_model, X_test, y_test, "DL", "DNN")
    print(accuracy_score)

















