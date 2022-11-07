import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

MAX_SIZE = 100
NUM_WORDS = 10000
CHUNKSIZE = 100000
NUM_EPOCHS = 10
EMBEDDING_DIM = 16


# Just to test the setup
def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


def tokenizer_func(data_rating, data_review):
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(data_review)

    word_index = tokenizer.word_index

    train_labels = data_rating.iloc[:int(len(data_rating) / 2)]
    train_examples = data_review.iloc[int(len(data_review) / 2):]
    test_examples = data_review.iloc[:int(len(data_review) / 2)]
    test_labels = data_rating.iloc[int(len(data_rating) / 2):]

    sequences_train = tokenizer.texts_to_sequences(train_examples)
    sequences_test = tokenizer.texts_to_sequences(test_examples)

    padded_train = pad_sequences(sequences_train, padding='post', truncating='post', maxlen=MAX_SIZE)
    padded_test = pad_sequences(sequences_test, padding='post', truncating='post', maxlen=MAX_SIZE)

    return np.array(padded_train), np.array(padded_test), np.array(train_labels), np.array(test_labels)


def model_using_padded(padded_train, padded_test, train_labels, test_label):
    pass


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
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
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

    for chunk in pd.read_csv('../kaggle/input/goodreads_train.csv', sep=',', header=0, chunksize=CHUNKSIZE):
        data_rating = chunk["rating"]
        data_review = chunk["review_text"]
        padded_train, padded_test, train_labels, test_labels = tokenizer_func(data_rating, data_review)
        sarcasm_prediction = sarcasm_model.predict(padded_train)
        sarcasm_df = pd.DataFrame(sarcasm_prediction, columns=["predict"])
        # TODO: this line can be use to take the predicted sarcasm score of a sentence.
        # for temporarity in sarcasm_df["predict"]:

        # temp = sarcasm_df.loc[2, "predict"]
