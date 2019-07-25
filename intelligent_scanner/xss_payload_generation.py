#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a demo of generating xss payload by using LSTM

"""
import os
import logging
import warnings
import numpy as np
from tensorflow import keras

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)


def load_file(file_path):
    """Load data file

    Args:
        file_path: string, the path of file

    Returns:
        texts: list of string, all the texts of file
    """

    with open(file_path) as file:

        # Store all texts
        texts = []

        logging.info("Load file successfully")
        for line in file.readlines():

            # Remove \n and \t
            line = line.strip('\n')
            line = line.strip('\t')
            texts.append(line)

    return texts


def get_sequences(texts, seq_length, step):
    """

    Args:
        max_len:

    Returns:

    """

    # Store all sampled sequence
    sequences = []

    # Store targets/labels corresponding to sequences
    targets = []

    for text in texts:

        # when the length of text <= max_len, then a sequence can not be got
        if len(text) <= seq_length:

            logging.info("Current text is invalid")
            continue

        else:

            for i in range(0, len(text)-seq_length, step):

                # Get a sequence from current text
                sequences.append(text[i: i + seq_length])

                # Get its target (next char)
                targets.append(text[i + seq_length])

            logging.info("Successfully get sequences from current text")

    return sequences, targets


def get_char_dict(texts):
    """

    Args:
        texts:

    Returns:

    """

    # Get the char set of each text
    texts = (set(text) for text in texts)

    # Get the entire unique char
    chars = sorted(set.union(*texts))

    # Get the char dict
    char_dict = dict((char, chars.index(char)) for char in chars)

    return char_dict


def get_key(dict, value):
    """

    Args:
        char_dict:

    Returns:

    """

    return [k for k, v in dict.items() if v == value][0]


def get_dataset(sequences, targets, char_dict, seq_length):
    """Get the dataset for training

    Args:
        sequences:
        target:

    Returns:

    """

    # Construct X, which is a tensor, shape (n_sequences, max_len, n_char)
    X = np.zeros((len(sequences), seq_length, len(char_dict)), dtype=np.bool)

    # Construct Y, which is a matrix, shape (n_sequences, n_char)
    Y = np.zeros((len(sequences), len(char_dict)), dtype=np.bool)

    for sequence_index, sequence in enumerate(sequences):

        for char_index, char in enumerate(sequence):

            # The location corresponding to current char is set to 1
            X[sequence_index, char_index, char_dict[char]] = 1

        # The location corresponding to current target is set to 1
        Y[sequence_index, char_dict[targets[sequence_index]]] = 1

    return X, Y


def lstm(timesteps, input_dim):
    """Build LSTM model.

    Args:
        timesteps: int, the length of sequence
        input_dim: int, the number of unique chars

    Returns:
        The built LSTM model
    """

    # Build LSTM network structure
    model = keras.Sequential([
        keras.layers.LSTM(units=128, input_shape=(timesteps, input_dim), dropout=0.2, return_sequences=True),
        keras.layers.LSTM(units=128, dropout=0.2),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=input_dim, activation="softmax")
    ])
    logging.info("Build LSTM model successfully.")

    # Compole the LSTM model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    logging.info("Compile LSTM model successfully.")

    return model


def sample(preds, temperatue=1.0):
    """Sample next char

    Args:
        preds: array, shape (input_dim, ), the predictions of LSTM network
        temperatue: float, control the generated probability distribution

    Returns:
        the index of the char with highest probability
    """

    # Transform to float type
    preds = np.array(preds).astype('float64')

    # Generate the probability distribution of this predictions
    preds = np.log(preds) / temperatue

    # Exponential transformation and normalization
    preds_exp = np.exp(preds)
    preds = preds_exp / np.sum(preds_exp)

    # Get the final probability from the generated distribution
    probas = np.random.multinomial(n=1, pvals=preds, size=1)

    # Get the index of the char with highest probability
    high_prob_index = np.argmax(probas)

    return high_prob_index


def train_model(model, X, Y, epochs, batch_size):
    """Train the model

    Args:
        model: the model that need to be trained
        X: array, shape (n_samples, timestpes, input_dim)
        Y: array, shape (n_samples, input_dim)
        epochs: int, the number of epoch
        batch_size: int, the size of batch

    Returns:
        the trained model
    """

    logging.info("Begin training the LSTM model")
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)
    logging.info("Training LSTM model successfully")

    return model


def generate_sequence(trained_model, seed_sequence, seq_generated_length, seq_length, char_dict, temperature):
    """

    Args:
        trained_model: the trained model
        seed_sequence: string, the seed for generating new sequence
        seq_generated_length: int, the length of the generated sequence
        seq_length: int, the length of the seed
        char_dict: dict, store all unique char
        temperature: float, control the generated probability distribution

    Returns:

    """

    seq_generated = ''

    for i in range(seq_generated_length):

        sequence = np.zeros((1, seq_length, len(char_dict)))

        for char_index, char in enumerate(seed_sequence):

            sequence[0, char_index, char_dict[char]] = 1

        preds = trained_model.predict(sequence, verbose=1)[0]
        next_index = sample(preds=preds, temperatue=temperature)
        next_char = get_key(char_dict, next_index)

        seed_sequence = seed_sequence + next_char
        seed_sequence = seed_sequence[1:]

        seq_generated += next_char

    return seq_generated


if __name__ == "__main__":

    # Set the path of data file
    file_path = "data/xss.txt"

    # Set the length of each sequence the step for sampling
    seq_length = 20
    step = 3

    # Get all texts
    texts = load_file(file_path)

    # Get unique char dict
    char_dict = get_char_dict(texts)

    # Get all sequences and their corresponding targets
    sequences, targets = get_sequences(texts, seq_length, step)

    # Get X and Y for training model
    X, Y = get_dataset(sequences, targets, char_dict, seq_length)

    # Train LSTM model
    lstm_model = lstm(timesteps=seq_length, input_dim=len(char_dict))
    trained_model = train_model(lstm_model, X, Y, epochs=10, batch_size=1000)

    # Generate a new sequence
    seed_text = texts[np.random.randint(0, len(texts)-1)]
    seed_sequence_start_index = np.random.choice(list(range(0, len(seed_text)-seq_length, 3)))
    seed_sequence = seed_text[seed_sequence_start_index:seed_sequence_start_index+seq_length]
    seq_generated = generate_sequence(trained_model, seed_sequence, seq_generated_length=20, seq_length=seq_length,
                                      char_dict=char_dict, temperature=0.5)
    print(seq_generated)




