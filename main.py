import data_preparation as dp
from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import custom_model
import numpy as np
import os
import keras_model as km
 

if __name__ == "__main__":
    sources = [
    ("./data/original/plates.csv", "./data/original/photos/"),
    ("./data/original/flipped_plates.csv", "./data/original/flipped_photos/"),
    ("./data/original/noise_plates.csv", "./data/original/noise_photos/"),
    ("./data/original/flipped_noise_plates.csv", "./data/original/flipped_noise_photos/")
    ]

    img_width = 320
    img_height = 200

    X_train, Y_train, X_test, Y_test, X_val, Y_val, original_train, original_test, original_val, filename_train, filename_test, filename_val = dp.load_data(sources, img_width, img_height)

    X = X_train
    Y = Y_train
    original_size = original_train
    filename_train = filename_train[0:2]

    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)

    # # Initialize the CNN model    
    # cnn = custom_model.CNN()

    # # Add layers to model
    # cnn.add_layer(Conv(8, 3, 1))
    # cnn.add_layer(ReLU())
    # cnn.add_layer(MaxPool(2))
    # cnn.add_layer(Conv(16, 3, 8))
    # cnn.add_layer(ReLU())
    # cnn.add_layer(MaxPool(2))
    # cnn.add_layer(Flatten())
    # cnn.add_layer(Dense(59904, 4))

    # cnn.print_model_structure()
    # cnn.summary(X.shape)
    
    # loss_fn = MSELoss()

    # loss_per_epochs, avg_accuracy, rescaled_bboxs_predicted_list, rescaled_bboxs_original_list = cnn.train(X, Y, original_size, 8, loss_fn, 2, 0.005)

    # print(rescaled_bboxs_predicted_list)
    # print(rescaled_bboxs_original_list)

    # keras_model = km.build_keras_cnn(img_height, img_width)

    # history = keras_model.fit(X, Y, batch_size=16, epochs=50)

    # y_predicted = keras_model.predict(X)

    # rescaled_predicted = dp.rescale_bbox(y_predicted, original_size)
    # rescaled_original = dp.rescale_bbox(Y, original_size)

    # print(rescaled_predicted)
    # print(rescaled_original)