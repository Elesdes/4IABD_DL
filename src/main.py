import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

MAX_SIZE = 400 # 783
NUM_WORDS = 254 # 7500
CHUNKSIZE = 10000
EPOCHS = 10
BATCH_SIZE = 1024
LOG_DIR = "tensorboard"
DATA_DIR = "../kaggle/input/goodreads_train.csv"
OOV = 1
SEED = 1000


# Just to test the setup
def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def tokenizer_func(data_rating, data_review):
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV)
    tokenizer.fit_on_texts(data_review)

    word_index = tokenizer.word_index

    train_labels = data_rating[math.floor(int(len(data_rating) / 8)):]
    train_examples = data_review[math.floor(int(len(data_review) / 8)):]
    test_examples = data_review[:math.floor(int(len(data_review) / 8))]
    test_labels = data_rating[:math.floor(int(len(data_rating) / 8))]

    sequences_train = tokenizer.texts_to_sequences(train_examples)
    sequences_test = tokenizer.texts_to_sequences(test_examples)

    padded_train = pad_sequences(sequences_train, padding='post', truncating='post', maxlen=MAX_SIZE)
    padded_test = pad_sequences(sequences_test, padding='post', truncating='post', maxlen=MAX_SIZE)

    return np.array(padded_train), np.array(padded_test), np.array(train_labels), np.array(test_labels)


def model_start(padded_train, padded_test, train_labels, test_labels, model, last_epochs):
    padded_train = padded_train / NUM_WORDS
    padded_test = padded_test / NUM_WORDS

    train_labels = tf.keras.utils.to_categorical(train_labels, 6)
    test_labels = tf.keras.utils.to_categorical(test_labels, 6)

    padded_train = np.expand_dims(padded_train, -1)
    padded_test = np.expand_dims(padded_test, -1)

    model.fit(
        padded_train,  # .values.astype(np.float32)
        train_labels,
        callbacks=[tf.keras.callbacks.TensorBoard(LOG_DIR + "/trash/")],
        initial_epoch=last_epochs,
        epochs=EPOCHS + last_epochs,
        validation_data=(padded_test, test_labels),
        verbose=1)
    # model.fit(padded_train, train_labels, epochs=EPOCHS, validation_data=(padded_test, test_labels))

    # callbacks=[tf.keras.callbacks.TensorBoard(LOG_DIR + "/TEST/")],
    return model


def set_model():
    # CNN
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(6,
                                    activation=tf.keras.activations.softmax))  # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.1),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    return model


def get_dataset():
    return tf.data.experimental.make_csv_dataset(DATA_DIR,
                                                 batch_size=CHUNKSIZE,
                                                 select_columns=[3, 4],
                                                 label_name='rating',
                                                 shuffle=False,
                                                 ignore_errors=True)
if __name__ == '__main__':
    setup()
    tf.keras.utils.set_random_seed(SEED)
    # sarcasm_model = tf.keras.models.load_model('../algorithms/Sarcasm')
    model = set_model()
    data_batch = get_dataset()
    last_epochs = 0
    for batch, label in data_batch.take(-1):
        for key, value in batch.items():
            padded_train, padded_test, train_labels, test_labels = tokenizer_func(np.array(label).astype('str'), np.array(value).astype('str'))
            # sarcasm_prediction_train = sarcasm_model.predict(padded_train)
            # sarcasm_prediction_test = sarcasm_model.predict(padded_test)
            # padded_train = np.concatenate((padded_train, np.array(sarcasm_prediction_train.flatten())[:, None]), axis=1)
            # padded_test = np.concatenate((padded_test, np.array(sarcasm_prediction_test.flatten())[:, None]), axis=1)
            padded_train = np.reshape(padded_train, (1 - math.floor(int(len(label) / 8)), int(math.sqrt(MAX_SIZE + 1)), int(math.sqrt(MAX_SIZE + 1))))
            padded_test = np.reshape(padded_test, (math.floor(int(len(label) / 8)), int(math.sqrt(MAX_SIZE + 1)), int(math.sqrt(MAX_SIZE + 1))))
            model = model_start(padded_train, padded_test, train_labels, test_labels, model, last_epochs)
            last_epochs += EPOCHS

    model.save("../algorithms/DL/Test")

    """
    for chunk in pd.read_csv('../../kaggle/input/goodreads_train.csv', sep=',', header=0, chunksize=CHUNKSIZE):
        data_rating = chunk["rating"]
        data_review = chunk["review_text"]
        padded_train, padded_test, train_labels, test_labels = tokenizer_func(data_rating, data_review)
        sarcasm_prediction_train = sarcasm_model.predict(padded_train)
        sarcasm_prediction_test = sarcasm_model.predict(padded_test)

        padded_train = np.concatenate((padded_train, np.array(sarcasm_prediction_train.flatten())[:, None]), axis=1)
        padded_test = np.concatenate((padded_test, np.array(sarcasm_prediction_test.flatten())[:, None]), axis=1)

        padded_train = np.reshape(padded_train, (1 - math.floor(int(len(data_rating) / 8)), 28, 28))
        padded_test = np.reshape(padded_test, (math.floor(int(len(data_rating) / 8)), 28, 28))
        model = model_start(padded_train, padded_test, train_labels, test_labels, model)

    model.save("../algorithms/DL/Test")
    """
