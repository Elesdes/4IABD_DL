import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers, regularizers
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

MAX_SIZE = 784
NUM_WORDS = 20000
BATCH_SIZE_TRAIN = 9000  # 90000
BATCH_SIZE_TEST = 2250  # 22500
STEPS_PER_EPOCH = math.ceil(720000 / BATCH_SIZE_TRAIN)
EPOCHS = 20
EMBEDDING_DIMS = 64
NUM_HEADS = 2
FF_DIMS = 32
L2_COEF = 0.00001
LOG_DIR = "tensorboard"
DATA_DIR = "../kaggle/input/train_chunked.csv" # 720000 lines
TEST_DIR = "../kaggle/input/test_chunked.csv" # 180000 lines
OOV = 1
SEED = 1000


def setup():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


@tf.function
def get_dataset(dir, batch_s):
    return tf.data.experimental.make_csv_dataset(dir,
                                                 batch_size=batch_s,
                                                 select_columns=[3, 4],
                                                 label_name='rating',
                                                 num_epochs=EPOCHS,
                                                 shuffle=True,
                                                 shuffle_seed=SEED,
                                                 ignore_errors=False)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


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
    model.add(tf.keras.layers.Reshape((int(math.sqrt(MAX_SIZE)), int(math.sqrt(MAX_SIZE)), -1), input_shape=(None, MAX_SIZE)))
    return model


def set_model(model):
    # CNN
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.tanh, padding='same', kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu, kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))  # tf.keras.activations.tanh
    model.add(tf.keras.layers.Dense(6, activation=tf.keras.activations.softmax, kernel_regularizer=regularizers.l2(L2_COEF), bias_regularizer=regularizers.l2(L2_COEF)))  # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))

    model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    return model


def set_model_transformers(value):
    inputs = layers.Input(shape=(1,), dtype=tf.string) # layers.Input(shape=(MAX_SIZE,))
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=NUM_WORDS,
        standardize=None,
        output_mode='int',
        output_sequence_length=MAX_SIZE)
    vectorize_layer.adapt(np.array(value).astype('str'))
    x = vectorize_layer(inputs)
    embedding_layer = TokenAndPositionEmbedding(MAX_SIZE, NUM_WORDS, EMBEDDING_DIMS)
    x = embedding_layer(x)
    transformer_block = TransformerBlock(EMBEDDING_DIMS, NUM_HEADS, FF_DIMS)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(6, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9), loss=tf.keras.losses.categorical_crossentropy, metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def model_start(value_train, label_train, value_test, label_test, model):
    label_train = tf.keras.utils.to_categorical(label_train, 6, dtype='int64')
    label_test = tf.keras.utils.to_categorical(label_test, 6, dtype='int64')
    model.fit(
        value_train,
        label_train,
        callbacks=[tf.keras.callbacks.TensorBoard(LOG_DIR + "/Transformers_test/")],
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=(value_test, label_test),
        validation_steps=STEPS_PER_EPOCH,
        verbose=1)
    return model


if __name__ == '__main__':
    setup()
    steps = 1
    epoch = 1
    data_batch = get_dataset(DATA_DIR, BATCH_SIZE_TRAIN)
    test_batch = get_dataset(TEST_DIR, BATCH_SIZE_TEST)
    for batch, label in data_batch.take(1):
        for key, value in batch.items():
            model = set_model_transformers(value)

    data_batch = get_dataset(DATA_DIR, BATCH_SIZE_TRAIN)
    for batch_train, label_train in data_batch.take(STEPS_PER_EPOCH):
        if steps == 9:
            steps = 1
            epoch += 1
        print("Epoch: ", epoch, "Step: ", steps)
        steps += 1
        for batch_test, label_test in test_batch.take(1):
            for key_train, value_train in batch_train.items():
                for key_test, value_test in batch_test.items():
                    model = model_start(value_train, label_train, value_test, label_test, model)


    sentence = ["This book definitely reminds me of 'The Giver' by Lois Lowry, but it's not like that's a bad thing. Although I did really enjoy the giver, I also enjoyed matched. The similarity doesn't bother me: they're both greatly written books, and thats what matters most. I like how the people in the world within matched are still aware of what's going on, enough for some of them to kinda get that something is wrong. I also love how this book has a romance!! I can't wait to get my hands on the next book!"]
    # 4
    predicted = model.predict(sentence)
    print(predicted)
    print(np.argmax(predicted, axis=1))
    model.save("algorithms/DL/Transformers_test")
