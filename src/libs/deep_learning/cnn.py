from typing import Any

import tensorflow as tf


def model_start(padded_train: Any = None, padded_test: Any = None,
                train_labels: Any = None, test_labels: Any = None,
                batch_size: int = None, epochs: int = None) -> None:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(42, activation=tf.keras.activations.tanh))
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.tanh))
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.sigmoid))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
                  loss=tf.keras.losses.mse,
                  batch_size=batch_size,
                  epoch=epochs)

    model.fit(padded_train, train_labels, validation_data=(padded_test, test_labels))
