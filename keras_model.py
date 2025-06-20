import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from bbox_accuracy import mean_iou


def build_keras_cnn(img_height, img_width, learning_rate):
    keras_cnn_model = Sequential()

    keras_cnn_model.add(Input(shape=(img_height, img_width, 1))) 
    keras_cnn_model.add(Conv2D(8, (3, 3), padding="valid", activation='relu', use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1), bias_initializer="zeros"))
    keras_cnn_model.add(MaxPooling2D((2, 2)))

    keras_cnn_model.add(Conv2D(16, (3, 3), padding="valid", activation='relu', use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1), bias_initializer="zeros"))
    keras_cnn_model.add(MaxPooling2D((2, 2)))

    keras_cnn_model.add(Flatten())
    keras_cnn_model.add(Dense(16, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    bias_initializer='zeros'))
    keras_cnn_model.add(Dense(8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    bias_initializer='zeros'))
    keras_cnn_model.add(Dense(4, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    bias_initializer='zeros'))
    
    optimizer = Adam(learning_rate=learning_rate)

    keras_cnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[mean_iou])

    keras_cnn_model.summary()
    return keras_cnn_model


