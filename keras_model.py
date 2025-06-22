import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from bbox_accuracy import mean_iou
from keras.callbacks import ModelCheckpoint, EarlyStopping


'''
Structure of custom CNN model:
    1. Input: image - array of shape (img_height, img_width, 1)

    2. First Convolutional layer: filter (3, 3, 1), number of kernels = 8, padding = 0, stride = 1
    3. ReLu Activation layer
    4. MaxPooling layer: stride = 2, size 2x2

    5. Second Convolution layer, number of kernels = 16, padding = 0, stride = 1
    6. Second Relu Actibvation layer
    7. Second MaxPooling layer: stride = 2, size 2x2

    8. Flattening layer

    9. Fully Connected (Dense) layers x3

    Loss function = MSE
    Accuracy: IoU
'''


def build_keras_model(img_height, img_width, learning_rate, min_delta, patience, filename):
    # Sequential groups a linear stack of layers into a Model
    keras_cnn_model = Sequential()

    # Adds layer instances
    keras_cnn_model.add(Input(shape=(img_height, img_width, 1))) 
    keras_cnn_model.add(Conv2D(8, (3, 3), padding="valid", activation='relu', use_bias=True, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer="zeros"))
    keras_cnn_model.add(MaxPooling2D((2, 2)))

    keras_cnn_model.add(Conv2D(16, (3, 3), padding="valid", activation='relu', use_bias=True, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer="zeros"))
    keras_cnn_model.add(MaxPooling2D((2, 2)))

    keras_cnn_model.add(Flatten())

    keras_cnn_model.add(Dense(8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    bias_initializer='zeros'))
    keras_cnn_model.add(Dense(8, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    bias_initializer='zeros'))
    keras_cnn_model.add(Dense(4, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
    bias_initializer='zeros'))
    
    # Gradient descent (with momentum) optimizer
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    # Configures the model for training, loss = MSE, metrics = IoU accuracy
    keras_cnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[mean_iou])

    # Callback to save the Keras model, selected epoch with max val_mean_iou
    checkpoint = ModelCheckpoint(filename, monitor='val_mean_iou', mode='max', save_best_only=True)

    # Stop training when a monitored metric has stopped improving, monitored metric is val_mean_iou
    early_stop = EarlyStopping(monitor='val_mean_iou', mode='max', min_delta=min_delta, patience=patience, restore_best_weights=True, verbose=1)

    # Print summary of model
    keras_cnn_model.summary()

    return keras_cnn_model, checkpoint, early_stop


