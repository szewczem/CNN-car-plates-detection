import data_preparation as dp
from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import model
import numpy as np
import os
 

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, X_val, Y_val = dp.load_data()

    X = X_train[0:3]
    y_original = Y_train[0:3]

    # Initialize the CNN model    
    model = model.CNN()

    # Add layers to model
    model.add_layer(Conv(8, 3, 1))
    model.add_layer(ReLU())
    model.add_layer(MaxPool(2))
    model.add_layer(Conv(16, 3, 8))
    model.add_layer(ReLU())
    model.add_layer(MaxPool(2))
    model.add_layer(Flatten())
    model.add_layer(Dense(247744, 4))

    model.print_model_structure()
    
    loss_fn = MSELoss()
