import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorboard
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

MAX_SIZE = 783
NUM_WORDS = 1000
CHUNKSIZE = 100000
NUM_EPOCHS = 10
EMBEDDING_DIM = 16
EPOCHS = 10
BATCH_SIZE = 512
LOG_DIR = "src/tensorboard"
OOV = 0
SARCASM_TRAINING_SIZE = 20000


# Just to test the setup
def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


def tokenizer_func(data_rating, data_review):
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV)
    tokenizer.fit_on_texts(data_review)

    word_index = tokenizer.word_index

    train_labels = data_rating.iloc[math.floor(int(len(data_rating) / 8)):]
    train_examples = data_review.iloc[math.floor(int(len(data_review) / 8)):]
    test_examples = data_review.iloc[:math.floor(int(len(data_review) / 8))]
    test_labels = data_rating.iloc[:math.floor(int(len(data_rating) / 8))]

    sequences_train = tokenizer.texts_to_sequences(train_examples)
    sequences_test = tokenizer.texts_to_sequences(test_examples)

    padded_train = pad_sequences(sequences_train, padding='post', truncating='post', maxlen=MAX_SIZE)
    padded_test = pad_sequences(sequences_test, padding='post', truncating='post', maxlen=MAX_SIZE)

    return np.array(padded_train), np.array(padded_test), np.array(train_labels), np.array(test_labels)


def model_start(padded_train, padded_test, train_labels, test_labels, model):
    padded_train = padded_train / 255.0
    padded_test = padded_test / 255.0

    train_labels = tf.keras.utils.to_categorical(train_labels, 6)
    test_labels = tf.keras.utils.to_categorical(test_labels, 6)

    padded_train = np.expand_dims(padded_train, -1)
    padded_test = np.expand_dims(padded_test, -1)

    model.predict(padded_train)
    model.summary()
    history = model.fit(padded_train, train_labels,
              batch_size=BATCH_SIZE,
              callbacks=[tf.keras.callbacks.TensorBoard(LOG_DIR + "/TEST/")],
              epochs=EPOCHS, validation_data=(padded_test, test_labels))

    return model


def set_model():
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

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)) # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)) # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(6, activation=tf.keras.activations.softmax)) # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def define_sarcasm():
    data = pd.read_json('kaggle/input/sarcasm.json', lines=True)
    # iterating through the json data and loading the requisite values into our python lists
    sentences = data['headline']
    labels = data['is_sarcastic']


    training_sentences = sentences[0:SARCASM_TRAINING_SIZE]
    testing_sentences = sentences[SARCASM_TRAINING_SIZE:]

    training_labels = labels[0:SARCASM_TRAINING_SIZE]
    testing_labels = labels[SARCASM_TRAINING_SIZE:]
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV)
    # fitting tokenizer only to training set
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    # creating training sequences and padding them
    traning_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(traning_sequences, maxlen=MAX_SIZE,
                                    padding='post',
                                    truncating='post',
                                    )

    # creating  testing sequences and padding them using same tokenizer
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=MAX_SIZE,
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
        tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SIZE),
        # Global Average pooling is similar to adding up vectors in this case
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(training_padded, training_labels, epochs=NUM_EPOCHS,
                        validation_data=(testing_padded, testing_labels))

    return model


if __name__ == '__main__':
    setup()
    sarcasm_model = define_sarcasm()
    model = set_model()

    for chunk in pd.read_csv('kaggle/input/goodreads_train.csv', sep=',', header=0, chunksize=CHUNKSIZE):
        data_rating = chunk["rating"]
        data_review = chunk["review_text"]
        padded_train, padded_test, train_labels, test_labels = tokenizer_func(data_rating, data_review)
        sarcasm_prediction_train = sarcasm_model.predict(padded_train)
        sarcasm_prediction_test = sarcasm_model.predict(padded_test)

        padded_train = np.concatenate((padded_train, np.array(sarcasm_prediction_train.flatten())[:, None]), axis=1)
        padded_test = np.concatenate((padded_test, np.array(sarcasm_prediction_test.flatten())[:, None]), axis=1)

        padded_train = np.reshape(padded_train, (1 - math.floor(int(len(data_rating) / 8)), 28, 28))
        padded_test = np.reshape(padded_test, (math.floor(int(len(data_rating) / 8)), 28, 28))
        # padded_train = np.expand_dims(padded_train, -1)
        # padded_test = np.expand_dims(padded_test, -1)
        model = model_start(padded_train, padded_test, train_labels, test_labels, model)
