import data_preparation as dp
from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import custom_model
import numpy as np
import os
import keras_model as km


sources = [
    ("./data/original/plates.csv", "./data/original/photos/"),
    ("./data/original/flipped_plates.csv", "./data/original/flipped_photos/"),
    ("./data/original/noise_plates.csv", "./data/original/noise_photos/"),
    ("./data/original/flipped_noise_plates.csv", "./data/original/flipped_noise_photos/")
    ]

img_width = 320
img_height = 200

# Training parameters
batch_size = 32
epochs = 10
learning_rate = 0.05
min_delta = 0.01
patience = 15
filename = "./saved_models/custom_model_parameters.npz"
 

# ==================== EXECUTABLE ====================
def main(sources, img_width, img_height, batch_size, epochs, learning_rate, min_delta, patience, filename):
    X_train, Y_train, X_test, Y_test, X_val, Y_val, true_size_train, true_size_test, true_size_val, filename_train, filename_test, filename_val = dp.load_data(sources, img_width, img_height)
    print()

    X = X_train[0:2]
    Y = Y_train[0:2]
    true_size_train = true_size_train[0:2]
    filename_train = filename_train[0:2]

    X_val = X_val[:2]
    Y_val = Y_val[:2]
    true_size_val = true_size_val[:2]
    filename_val = filename_val[:2]

    # Initialize the CNN model    
    cnn = custom_model.CNN()
    print()

    # Add layers to model
    cnn.add_layer(Conv(8, 3, 1))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool(2))

    cnn.add_layer(Conv(16, 3, 8))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool(2))

    cnn.add_layer(Flatten())

    cnn.add_layer(Dense(59904, 8))
    cnn.add_layer(Dense(8, 8))
    cnn.add_layer(Dense(8, 4))
    print()

    # Print CNN info
    cnn.print_model_structure()
    cnn.summary(X.shape)
    print()

    # Defined loss function
    loss_fn = MSELoss()
    print()
    
    # Training with validation
    loss_per_epochs_train, loss_per_epochs_val, accuracy_per_epoch_train, accuracy_per_epoch_val, rescaled_bboxs_pred_train, rescaled_bboxs_true_train, rescaled_bboxs_pred_val, rescaled_bboxs_true_val = cnn.train(X, Y, true_size_train, X_val, Y_val, true_size_val, batch_size, loss_fn, epochs, learning_rate, min_delta, patience, filename=filename)
    print()

    # Example of rescaled bbox values for prdiction and ground truth (original data) for the training set
    print("\nExample of rescaled bbox values for prdiction and ground truth (original data) for the training set:")
    print(rescaled_bboxs_pred_train[:1])
    print(rescaled_bboxs_true_train[:1])

    # Example of rescaled bbox values for prdiction and ground truth (original data) for the validation set
    print("\nExample of rescaled bbox values for prdiction and ground truth (original data) for the validation set:")
    print(rescaled_bboxs_pred_val[:1])
    print(rescaled_bboxs_true_val[:1])
    print()

    # Initialization of second CNN model
    cnn2 = custom_model.CNN()
    print()

    # Using the same structure as the training model
    cnn2.add_layer(Conv(8, 3, 1))
    cnn2.add_layer(ReLU())
    cnn2.add_layer(MaxPool(2))

    cnn2.add_layer(Conv(16, 3, 8))
    cnn2.add_layer(ReLU())
    cnn2.add_layer(MaxPool(2))

    cnn2.add_layer(Flatten())
    cnn2.add_layer(Dense(59904, 8))
    cnn2.add_layer(Dense(8, 8))
    cnn2.add_layer(Dense(8, 4))

    # Setting the parameters learned from training the first model
    cnn2.set_parameters(filename)
    print()

    X_test = X_test[:6]
    Y_test = Y_test[:6]
    true_size_test = true_size_test[:6]
    filename_test = filename_test[:6]

    # Testing process, checking the model loss and model accuracy on the testing set
    test_total_loss, test_avg_iou, rescaled_bboxs_pred_test, rescaled_bboxs_true_test = test_total_loss, test_avg_iou, rescaled_test_pred, rescaled_test_true = cnn2.test(X_test, Y_test, true_size_test, batch_size, loss_fn, filename)
    print()

    # Example of rescaled bbox values for prdiction and ground truth (original data) for the test set
    print("\nExample of rescaled bbox values for prdiction and ground truth (original data) for the test set:")
    print(rescaled_bboxs_pred_test[:1])
    print(rescaled_bboxs_true_test[:1])
   

if __name__ == "__main__":
    main(sources, img_width, img_height, batch_size, epochs, learning_rate, min_delta, patience, filename)