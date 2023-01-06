import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

MAX_SIZE = 784
NUM_WORDS = 20000
BATCH_SIZE = 100000
STEPS_PER_EPOCH = math.ceil(900000 / BATCH_SIZE)
EPOCHS = 20
EMBEDDING_DIMS = 16
LOG_DIR = "tensorboard"
DATA_DIR = "../kaggle/input/goodreads_train.csv"
OOV = 1
SEED = 1000


def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def get_dataset():
    return tf.data.experimental.make_csv_dataset(DATA_DIR,
                                                 batch_size=BATCH_SIZE,
                                                 select_columns=[3, 4],
                                                 label_name='rating',
                                                 shuffle=True,
                                                 ignore_errors=True)


def get_preprocessing_model(value):
    # Create the layer.
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=NUM_WORDS,
        standardize=None,
        output_mode='int',
        output_sequence_length=MAX_SIZE)
    vectorize_layer.adapt(np.array(value).astype('str'))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    # normalizer_layer = layers.Normalization()
    # model.add(normalizer_layer)
    model.add(tf.keras.layers.Embedding(NUM_WORDS + 1, EMBEDDING_DIMS))
    model.add(tf.keras.layers.Reshape((28, 28, -1), input_shape=(None, 784)))
    return model


def set_model(model):
    # CNN
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.tanh, padding='same'))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(6,
                                    activation=tf.keras.activations.softmax))  # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.0001, momentum=0.9),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def model_start(value, label, model, last_epochs):
    train_labels = tf.keras.utils.to_categorical(label, 6)
    model.fit(
        value,
        train_labels,
        callbacks=[tf.keras.callbacks.TensorBoard(LOG_DIR + "/preprocessing_test_20_take_100000_batchsize_steps_per_epoch_on/")],
        # initial_epoch=last_epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS, # + last_epochs,
        validation_split=0.2,
        verbose=1)

    return model


if __name__ == '__main__':
    setup()
    data_batch = get_dataset()
    last_epochs = 0
    for batch, label in data_batch.take(1):
        for key, value in batch.items():
            model = get_preprocessing_model(value)

    model = set_model(model)
    for batch, label in data_batch.take(1):
        for key, value in batch.items():
            model = model_start(value, label, model, last_epochs)
            last_epochs += EPOCHS

    model.save("algorithms/DL/20_take")
