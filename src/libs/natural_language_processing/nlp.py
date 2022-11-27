from typing import Any

from src.libs.utils.default import DefaultParameters

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from pandas import DataFrame


def tokenizer_func(data_rating: DataFrame = None, data_review: DataFrame = None,
                   nb_words: int = DefaultParameters.num_words, max_size: int = DefaultParameters.max_size) \
        -> tuple[np.array, np.array, np.array, np.array]:
    tokenizer = Tokenizer(num_words=nb_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data_review)

    word_index = tokenizer.word_index

    train_labels = data_rating.iloc[:int(len(data_rating) / 2)]
    train_examples = data_review.iloc[int(len(data_review) / 2):]
    test_examples = data_review.iloc[:int(len(data_review) / 2)]
    test_labels = data_rating.iloc[int(len(data_rating) / 2):]

    sequences_train = tokenizer.texts_to_sequences(train_examples)
    sequences_test = tokenizer.texts_to_sequences(test_examples)

    padded_train = pad_sequences(sequences_train, padding='post', truncating='post', maxlen=max_size)
    padded_test = pad_sequences(sequences_test, padding='post', truncating='post', maxlen=max_size)

    return np.array(padded_train), np.array(padded_test), np.array(train_labels), np.array(test_labels)
