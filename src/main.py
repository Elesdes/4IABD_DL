from src.libs.natural_language_processing.nlp import tokenizer_func
from src.libs.utils.default import DefaultParameters

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# Just to test the utils
def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


def define_sarcasm():
    data = pd.read_json('../kaggle/input/sarcasm.json', lines=True)
    # iterating through the json data and loading the requisite values into our python lists
    sentences = data['headline']
    labels = data['is_sarcastic']
    urls = data['article_link']
    training_size = 20000

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]

    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]
    tokenizer = Tokenizer(num_words=DefaultParameters.num_words, oov_token="<OOV>")
    # fitting tokenizer only to training set
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    # creating training sequences and padding them
    traning_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(traning_sequences, maxlen=DefaultParameters.max_size,
                                    padding='post',
                                    truncating='post',
                                    )

    # creating  testing sequences and padding them using same tokenizer
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=DefaultParameters.max_size,
                                   padding='post',
                                   truncating='post',
                                   )

    # converting all variables to numpy arrays, to be able to work with tf version 2
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    # creating a model for sentiment analysis
    model = tf.keras.Sequential([
        # addinging an Embedding layer for Neural Network to learn the vectors
        tf.keras.layers.Embedding(DefaultParameters.num_words, DefaultParameters.embedding_dim,
                                  input_length=DefaultParameters.max_size),
        # Global Average pooling is similar to adding up vectors in this case
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(training_padded, training_labels, epochs=DefaultParameters.epochs,
                        validation_data=(testing_padded, testing_labels))

    return model


if __name__ == '__main__':
    setup()
    sarcasm_model = define_sarcasm()

    for chunk in pd.read_csv('../kaggle/input/goodreads_train.csv', sep=',', header=0,
                             chunksize=DefaultParameters.chunk_size):
        data_rating = chunk["rating"]
        data_review = chunk["review_text"]
        padded_train, padded_test, train_labels, test_labels = tokenizer_func(data_rating, data_review)
        sarcasm_prediction_train = sarcasm_model.predict(padded_train)
        sarcasm_prediction_test = sarcasm_model.predict(padded_test)

        padded_train = np.concatenate((padded_train, np.array(sarcasm_prediction_train.flatten())[:, None]), axis=1)
        padded_test = np.concatenate((padded_test, np.array(sarcasm_prediction_test.flatten())[:, None]), axis=1)

        # model_start(padded_train, padded_test, train_labels, test_labels)

