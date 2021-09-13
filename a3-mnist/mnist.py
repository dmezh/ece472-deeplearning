#!/bin/env/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

    t_images = import_images('t10k-images-idx3-ubyte')
    t_labels = import_labels('t10k-labels-idx1-ubyte')

    BATCH_SIZE = 128

    images = np.expand_dims(images, axis=-1) # add batch axis
    labels = tf.keras.utils.to_categorical(labels, num_classes=10) # change from 0-9 to categorical ([0,..1,0])
    t_images = np.expand_dims(t_images, axis=-1)
    t_labels = tf.keras.utils.to_categorical(t_labels, num_classes=10)

    print("Training ...")

    # input -> Dropout -> Conv2D -> MaxPooling2D -> Dropout -> Conv2D -> MaxPooling2D -> Flatten -> Dropout -> Dense (output)
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.3, input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(28, (4,4), activation='relu', input_shape=(28, 28, 1), kernel_regularizer='l2', use_bias=True),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(52, (3,3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer='l2', use_bias=True),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer='l2', use_bias=True),
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()
    model.fit(images, labels, batch_size=BATCH_SIZE, epochs=5, validation_split=0.2)
    model.evaluate(t_images, t_labels)

if __name__ == "__main__":
    main()
