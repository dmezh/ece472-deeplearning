#!/bin/env/python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

import pickle

def import_images(filename):
    with open(filename, mode='rb') as images:

        header = np.fromfile(images, count=4, dtype='>u4')
        magic_number, image_count, row_count, col_count = header

        print(f'Magic number: {magic_number}, number of images: {image_count}, number of rows: {row_count}, number of columns: {col_count}')

        images_buf = np.fromfile(images, dtype='u1')

        return images_buf.reshape(image_count, row_count, col_count)

def import_labels(filename):
    with open(filename, mode='rb') as labels:
        
        header = np.fromfile(labels, count=2, dtype='>u4')
        magic_number, labels_count = header

        print(f'Magic number: {magic_number}, number of labels: {labels_count}')

        labels_buf = np.fromfile(labels, dtype='u1')

        return labels_buf.reshape(labels_count)


def main():
    images = import_images('train-images-idx3-ubyte')
    labels = import_labels('train-labels-idx1-ubyte')

    training_data = tf.data.Dataset.from_tensor_slices((images, labels))

    t_images = import_images('t10k-images-idx3-ubyte')
    t_labels = import_labels('t10k-labels-idx1-ubyte')

    test_data = tf.data.Dataset.from_tensor_slices((t_images, t_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_data = test_data.batch(500)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

    model.fit(training_data, epochs=12)
    model.evaluate(test_data)

if __name__ == "__main__":
    main()
