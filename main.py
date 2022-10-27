import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from numpy import genfromtxt
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd


# Just to test the setup
def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# Loading the training and testing data
def loading_data():
    data = pd.read_csv('kaggle/input/goodreads_train.csv', sep=',', header=0)
    data_rating = data["rating"]
    data_review = data["review_text"]
    """
    train_labels = data_rating.iloc[:int(len(data_rating)/2)]
    train_examples = data_review.iloc[int(len(data_review)/2):]
    test_examples = data_review.iloc[:int(len(data_review)/2)]
    test_labels = data_rating.iloc[int(len(data_rating)/2):]
    """

    print("Training entries: {}, test entries: {}".format(len(data_rating), len(data_review)))
    return data_rating, data_review


def tokenizer_func(data_rating, data_review):
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(data_review)

    word_index = tokenizer.word_index

    train_labels = data_rating.iloc[:int(len(data_rating) / 2)]
    train_examples = data_review.iloc[int(len(data_review) / 2):]
    test_examples = data_review.iloc[:int(len(data_review) / 2)]
    test_labels = data_rating.iloc[int(len(data_rating) / 2):]

    sequences_train = tokenizer.texts_to_sequences(train_examples)
    sequences_test = tokenizer.texts_to_sequences(test_examples)

    padded_train = pad_sequences(sequences_train, padding='post', truncating='post')
    padded_test = pad_sequences(sequences_test, padding='post', truncating='post')

    return padded_train, padded_test, train_labels, test_labels


def model_using_padded(padded_train, padded_test, train_labels, test_label):
    embedding_dim = 16

    # creating a model for sentiment analysis
    model = tf.keras.Sequential([
        # addinging an Embedding layer for Neural Network to learn the vectors
        tf.keras.layers.Embedding(1000, embedding_dim, input_length=100),
        # Global Average pooling is similar to adding up vectors in this case
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    num_epochs = 10

    history = model.fit(padded_train, train_labels, epochs=num_epochs,
                        validation_data=(padded_test, test_label))


# Use only for a plot_lib schema. Otherwise, use tensorboard
def plot_lib_print(epochs, loss, val_loss, acc, val_acc):
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    setup()
    data_rating, data_review = loading_data()
    padded_train, padded_test, train_labels, test_labels = tokenizer_func(data_rating, data_review)
    model_using_padded(padded_train, padded_test, train_labels, test_labels)
    # epochs, loss, val_loss, acc, val_acc = model_usage(train_examples, train_labels, test_examples, test_labels)
    # plot_lib_print(epochs, loss, val_loss, acc, val_acc)
