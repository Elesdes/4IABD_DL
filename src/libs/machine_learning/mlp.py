from src.libs.natural_language_processing.nlp import tokenizer_func
from src.libs.utils.default import DefaultParameters
from src.libs.utils.tensorboard import create_tensorboard_log_dir

import numpy as np
import pandas as pd
import tensorflow as tf


@create_tensorboard_log_dir(type_of_algorithm="ML",
                            algorithm_used="MLP")
def mlp(path: str = None):
    pass


if __name__ == "__main__":
    train_df = pd.read_csv('../../../kaggle/input/goodreads_train.csv')
    test_df = pd.read_csv('../../../kaggle/input/goodreads_test.csv')
    dfs = [train_df, test_df]
    for df in dfs:
        for chunk in pd.read_csv('../kaggle/input/goodreads_train.csv', sep=',', header=0,
                                 chunksize=DefaultParameters.chunk_size):
            data_rating = chunk["rating"]
            data_review = chunk["review_text"]
            padded_train, padded_test, train_labels, test_labels = tokenizer_func(data_rating, data_review)


