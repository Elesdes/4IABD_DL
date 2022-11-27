from typing import Any

from src.libs.natural_language_processing.nlp import tokenizer_func
from src.libs.utils.default import DefaultParameters
from src.libs.utils.tf_utils import create_tensorboard_log_dir

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf


@create_tensorboard_log_dir(type_of_algorithm="ML",
                            algorithm_used="MLP")
def mlp(x_train, y_train, path: str = None) -> Any:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(treshold=0.0))

    history = model.fit(
        x_train, y_train,
        epochs=DefaultParameters.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=path)
        ],
        batch_size=DefaultParameters.batch_size
    )


if __name__ == "__main__":
    train_df = pd.read_csv('../../../kaggle/input/goodreads_train.csv')
    test_df = pd.read_csv('../../../kaggle/input/goodreads_test.csv')
    input_dfs = [train_df, test_df]
    datasets = ["train_ds", "test_ds"]

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=DefaultParameters.num_words,
        output_sequence_length=DefaultParameters.max_size,
        standardize='lower_and_strip_punctuation'
    )

    for index, df in tqdm(enumerate(input_dfs), 0):
        dataset = [rows for rows in df[['rating', 'review_text']]]
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        vectorize_layer.adapt(dataset.batch(64))


