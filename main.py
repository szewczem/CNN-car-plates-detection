import data_preparation as dp
from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import model
import numpy as np
import os
 

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, X_val, Y_val, original_train, original_test, original_val, filename_train, filename_test, filename_val = dp.load_data(img_width=320, img_height=200)

    X = X_train[0:2]
    Y = Y_train[0:2]
    original_size = original_train[0:2]
    filename_train = filename_train[0:2]

    # Initialize the CNN model    
    cnn = model.CNN()

    # Add layers to model
    cnn.add_layer(Conv(8, 3, 1))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool(2))
    cnn.add_layer(Conv(16, 3, 8))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool(2))
    cnn.add_layer(Flatten())
    cnn.add_layer(Dense(59904, 4))

    cnn.print_model_structure()
    
    loss_fn = MSELoss()

    loss_per_epochs, avg_accuracy, rescaled_bboxs_predicted_list, rescaled_bboxs_original_list = cnn.train(X, Y, original_size, 16, loss_fn, 25, 0.01)

    print(rescaled_bboxs_predicted_list)
    print(rescaled_bboxs_original_list)