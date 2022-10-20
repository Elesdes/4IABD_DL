import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd


# Just to test the setup
def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# Loading the training and testing data
def loading_data():
    # train_data = genfromtxt('data/goodreads_train.csv', delimiter=',', skip_header=1)
    # train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
    #                                   batch_size=-1, as_supervised=True)
    # test_data = genfromtxt('data/goodreads_test.csv', delimiter=',', skip_header=1)

    data = pd.read_csv('data/goodreads_train.csv', sep=',', header=0)
    # test_data = pd.read_csv('data/goodreads_test.csv', sep=',', header=0)
    data_rating = data["rating"]
    data_review = data["review_text"]

    train_labels = data_rating.iloc[:int(len(data_rating)/2)]
    train_examples = data_review.iloc[int(len(data_review)/2):]
    test_examples = data_review.iloc[:int(len(data_review)/2)]
    test_labels = data_rating.iloc[int(len(data_rating)/2):]

    # train_examples, train_labels = tfds.as_numpy(train_data)
    # test_examples, test_labels = tfds.as_numpy(test_data)

    print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
    print(train_examples.values)
    print(train_labels.values)
    return train_examples, train_labels, test_examples, test_labels


# Setup Model and training + result. Might be changed in a near future
def model_usage(train_examples, train_labels, test_examples, test_labels):
    model = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

    x_val = train_examples[:10000]
    partial_x_train = train_examples[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    results = model.evaluate(test_examples, test_labels)
    print(results)

    history_dict = history.history
    history_dict.keys()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    return epochs, loss, val_loss, acc, val_acc


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
    train_examples, train_labels, test_examples, test_labels = loading_data()
    epochs, loss, val_loss, acc, val_acc = model_usage(train_examples, train_labels, test_examples, test_labels)
    plot_lib_print(epochs, loss, val_loss, acc, val_acc)
