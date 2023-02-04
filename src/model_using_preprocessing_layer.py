import collections
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers, regularizers
from typing import Any, Union

MAX_SIZE = [4096]  # 1024
NUM_WORDS = [20000]
BATCH_SIZE_TRAIN = 900000  # 90000 720000
BATCH_SIZE_TEST = 180000  # 22500
BATCH_SIZE = [1024]  # Train = 900000 | Test = 478033
STEPS_PER_EPOCH = math.ceil(720000 / BATCH_SIZE_TRAIN)
EPOCHS = [20]
EMBEDDING_DIMS = [32]
L2_COEF = [0.00001]
LOG_DIR = "tensorboard"
DATA_DIR = "../kaggle/input/goodreads_train.csv"  # 720000 lines
TEST_DIR = "../kaggle/input/test_chunked.csv"  # 180000 lines
OOV = 1
SEED = [1001]
INIT_SGD = [0.001]
MOMENTUM = [0.005]
DROPOUT = [0.00001]


def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def get_dataset(path: str,
                batch_size: int,
                seed: int,
                separator: Union[list[str], str] = ',',
                columns: list[str] = None) -> collections.OrderedDict:
    # Load data -> tensors
    dataset = tf.data.experimental.make_csv_dataset(
        path,
        batch_size=batch_size,
        field_delim=separator,
        select_columns=columns,
        prefetch_buffer_size=batch_size,
        shuffle=False,
        shuffle_seed=seed)
    # Get an iterator over the dataset

    iterator = dataset.as_numpy_iterator()
    return next(iterator)


def get_preprocessing_model(value,
                            num_words: int,
                            max_size: int,
                            embedding_dims: int):
    # Create the layer.
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=num_words,
        standardize=None,
        output_mode='int',
        output_sequence_length=max_size)
    vectorize_layer.adapt(np.array(value).astype('str'))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    # normalizer_layer = layers.Normalization()
    # model.add(normalizer_layer)
    model.add(tf.keras.layers.Embedding(num_words + 1, embedding_dims))
    model.add(
        tf.keras.layers.Reshape((int(math.sqrt(max_size)), int(math.sqrt(max_size)), -1), input_shape=(None, max_size)))
    return model


def normalize_data(x: np.ndarray, y: np.ndarray, batch_size: int) -> Any:
    y = tf.keras.utils.to_categorical(y, 6, dtype='int64')
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size)
    return dataset


def set_model(model, l2_coef, init_sgd, momentum, dropout_rate):
    # CNN
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same',
                                     kernel_regularizer=regularizers.l2(l2_coef),
                                     bias_regularizer=regularizers.l2(l2_coef)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same',
                                     kernel_regularizer=regularizers.l2(l2_coef),
                                     bias_regularizer=regularizers.l2(l2_coef)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same',
                                     kernel_regularizer=regularizers.l2(l2_coef),
                                     bias_regularizer=regularizers.l2(l2_coef)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same',
                                     kernel_regularizer=regularizers.l2(l2_coef),
                                     bias_regularizer=regularizers.l2(l2_coef)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPool2D())


    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizers.l2(l2_coef),
                              bias_regularizer=regularizers.l2(l2_coef)))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(
        tf.keras.layers.Dense(6, activation=tf.keras.activations.softmax, kernel_regularizer=regularizers.l2(l2_coef),
                              bias_regularizer=regularizers.l2(
                                  l2_coef)))  # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.SGD(init_sgd, momentum=momentum),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    return model


def model_start(model, data_batch, batch_size, epochs, max_size, num_words, embedding_dims, l2_coef, seed, init_sgd, momentum):
    for text_train, label_train in data_batch.take(batch_size):
        model.fit(text_train,
                  label_train,
                  epochs=epochs,
                  validation_split=0.2,
                  # validation_data=(test_batch),
                  callbacks=[tf.keras.callbacks.TensorBoard(LOG_DIR + f"/CNN"
                                                                      f"_MSize_{max_size}"
                                                                      f"_NWords_{num_words}"
                                                                      f"_BSize_{batch_size}"
                                                                      f"_Epochs_{epochs}"
                                                                      f"_EDims_{embedding_dims}"
                                                                      f"_L2Coef_{l2_coef}"
                                                                      f"_Seed_{seed}"
                                                                      f"_ISGD_{init_sgd}"
                                                                      f"_Mo_{momentum}"
                                                                      f"_D_{dropout_rate}"
                                                                      f"_Block_8-16/")])
    return model
    # label_train = tf.keras.utils.to_categorical(label_train, 6, dtype='int64')
    # label_test = tf.keras.utils.to_categorical(label_test, 6, dtype='int64')


if __name__ == '__main__':
    setup()
    to_skip = 1
    for max_size in MAX_SIZE:
        for num_words in NUM_WORDS:
            for batch_size in BATCH_SIZE:
                for epochs in EPOCHS:
                    for embedding_dims in EMBEDDING_DIMS:
                        for l2_coef in L2_COEF:
                            for seed in SEED:
                                for init_sgd in INIT_SGD:
                                    for momentum in MOMENTUM:
                                        for dropout_rate in DROPOUT:
                                            if to_skip:
                                                to_skip = 0
                                                pass
                                            data_batch = get_dataset(path=DATA_DIR, batch_size=BATCH_SIZE_TRAIN, seed=seed, columns=['review_text', 'rating'])
                                            # test_batch = get_dataset(path=TEST_DIR, batch_size=BATCH_SIZE_TEST, columns=['review_text', 'rating'], seed=seed)

                                            model = get_preprocessing_model(value=data_batch['review_text'], num_words=num_words, max_size=max_size, embedding_dims=embedding_dims)
                                            model = set_model(model, l2_coef=l2_coef, init_sgd=init_sgd, momentum=momentum, dropout_rate=dropout_rate)
                                            data_batch = get_dataset(path=DATA_DIR, batch_size=BATCH_SIZE_TRAIN, seed=seed, columns=['review_text', 'rating'])
                                            data_batch = normalize_data(data_batch['review_text'],
                                                                        data_batch['rating'],
                                                                        batch_size=BATCH_SIZE_TRAIN)
                                            model = model_start(model, data_batch, batch_size=batch_size, epochs=epochs, max_size=max_size, num_words=num_words, embedding_dims=embedding_dims, l2_coef=l2_coef, seed=seed, init_sgd=init_sgd, momentum=momentum)

                                            #model.save(f"algorithms/DL/CNN/64_64_images")
                                            #model.save(f"algorithms/DL/CNN/"
                                                       #f"MSize_{max_size}"
                                                       #f"_NWords_{num_words}"
                                                       #f"_BSize_{batch_size}"
                                                       #f"_Epochs_{epochs}"
                                                       #f"_EDims_{embedding_dims}"
                                                       #f"_L2Coef_{l2_coef}"
                                                       #f"_Seed_{seed}"
                                                       #f"_ISGD_{init_sgd}"
                                                       #f"_Mo_{momentum}"
                                                       #f"_Block_8-16-32")
