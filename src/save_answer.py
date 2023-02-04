import math

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv

BATCH_SIZE_TRAIN = 10000
EPOCHS = 20
DATA_DIR = "../kaggle/input/goodreads_test.csv"
OUTPUT_DIR = "../kaggle/output/answer_64_64_images.csv"
MODEL_DIR = "algorithms/DL/CNN/64_64_images"


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
                                                 select_columns=[2, 3],
                                                 label_name='review_id',
                                                 num_epochs=1,
                                                 shuffle=False,
                                                 ignore_errors=True)


if __name__ == '__main__':
    setup()
    data_batch = get_dataset(DATA_DIR, BATCH_SIZE_TRAIN)
    last_epochs = 0
    model = tf.keras.models.load_model(MODEL_DIR)
    with open(OUTPUT_DIR, 'w+', newline='\n') as fichiercsv:
        writer = csv.writer(fichiercsv)
        writer.writerow(['review_id', 'rating'])
        for batch_train, label_train in data_batch.take(math.ceil(478033/BATCH_SIZE_TRAIN)+1):
            print(last_epochs)
            last_epochs += 1
            for key_train, value_train in batch_train.items():
                predicted = model.predict(value_train.numpy().astype('unicode'))

                for i, j in zip(label_train.numpy().astype('unicode'), np.argmax(predicted, axis=1)):
                    writer.writerow([i, j])