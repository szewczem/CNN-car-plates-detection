import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from bbox_accuracy import mean_iou_keras
import data_preparation as dp
from itertools import product
import time
from keras.callbacks import EarlyStopping
import os


'''
Hyperparameters that can be tuned:
        learning rate -- step size for updating weights
        batch_size -- number of samples provided per batch
        number of epochs -- number of passes over entire dataset
        number of layers -- define how deep is the network
    For Convolutional layer:
        num_filters -- number of kernels (filters)
        filter_size -- dimension of kernel
        strides -- the rate at which to convolve (in px)
        padding -- include the edges of the img, change the output shape
        weights and biases -- initialization values
    For Activation layer:
        activation layer itself -- activation function can be changed
    For MaxPooling layer:
        filter_size -- window size for max-pooling
        strides -- the rate of step (in px), can skipping but shouldn't overlap
        padding -- include the edges of the img, change the output shape
    For Dense layer:
        outputs -- number of neurons
        activation function -- can be provided/change for adding nonlinearity
        weights and biases -- initialization values
    Other parameters:
        dropout -- percentage of neurons randomly set to 0, to prevent overfitting
        regularization technique -- regularization penalty on the output (activation), not on the weights
        momentum -- scalar value in SGD optimizers
        patience -- early stopping parameter to prevent training if no improvement is visible
    Data preparation:
        width, height, channels -- image dimensions

Chosen Parameters:
    The chosen parameter values closely correspond to the capabilities and design of the custom model.

    Tuning parameters:
        learning rate - too hight can cause parameters to explode, while too small may prevent the model from improving its accuracy
        batch_size -  smaller batch sizes may improve accuracy but will increase training time
        number of layers - convolutional layers are computationally expensive, so no additional Conv layers are added, the number of Dense layers can be adjusted slightly
        outputs_dense - number of neurons in dense layers, the first dense layer will be the bottleneck of CNN.

To reduce training time, the model should have as few trainable parameters as possible, < 500000.
'''


# Sets of .csv data and corresponding folder with images 
sources = [
    ("./data/original/plates.csv", "./data/original/photos/"),
    ("./data/original/flipped_plates.csv", "./data/original/flipped_photos/"),
    ("./data/original/noise_plates.csv", "./data/original/noise_photos/"),
    ("./data/original/flipped_noise_plates.csv", "./data/original/flipped_noise_photos/")
    ]

# Params
img_width = 320
img_height = 200
epochs = 15
valid_iou = 0.12

# Saving paths
models_path = "./saved_models"
model_params_file = models_path + "/best_keras_as_custom_model.keras"
top_models_result_file = models_path + "/top_models_result.txt"

# Searched params in hyperparameters tuning
learning_rates = [0.15, 0.13, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01]
batch_sizes = [32, 64]
# Checking combinations of numb_of_dense_layers and output_in_dense
dense_layer_config = [[4], [8], [4, 4], [8, 8], [4, 8], [8, 4], [4, 16], [8, 16], [4, 4, 16], [4, 8, 16], [8, 8, 16]]


# ==================== HYPERPARAMETERS TUNING ====================
def count_num_of_combinations(learning_rates, batch_sizes, dense_layer_config):
    num_of_combinations = len(learning_rates) * len(batch_sizes) * len(dense_layer_config)
    print(f"Number of tested combinations: {num_of_combinations}")


# Create the folder for outputs if not exists
def output_folder(folder):
    folder = os.path.dirname(model_params_file)
    if folder and not os.path.exists(folder):
        os.mkdir(folder)


def grid_search(learning_rates, batch_sizes, dense_layer_config, img_width, img_height, X_train, Y_train, X_val, Y_val, epochs, valid_iou):
    best_iou = -1
    best_params = None
    top = []

    # Try all combinations (grid search)
    for i, (lr, batch_size, dense_config) in enumerate(product(learning_rates, batch_sizes, dense_layer_config)):
        print(f"{i + 1}. learning_rate: {lr}, batch_size: {batch_size}, dense_layers: {len(dense_config)}, output_in_dense: {dense_config}")
        
        # Sequential groups a linear stack of layers into a Model
        model = Sequential()

        # Adds layer instances
        model.add(Input(shape=(img_height, img_width, 1)))

        model.add(Conv2D(8, (3, 3), padding="valid", activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer="zeros"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(16, (3, 3), padding="valid", activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer="zeros"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        # Try different Dense settings
        for output_dense in dense_config:
            model.add(Dense(output_dense, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), bias_initializer='zeros', use_bias=True))
        
        # Last Dense, provide 4 bbox coordinates outputs
        model.add(Dense(4, use_bias=True, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1),
        bias_initializer='zeros'))

        # Gradient descent (with momentum) optimizer
        optimizer = SGD(learning_rate=lr, momentum=0.9)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[mean_iou_keras])

        # Stop training when a monitored metric has stopped improving, monitored metric is val_mean_iou_keras
        early_stop = EarlyStopping(monitor='val_mean_iou_keras', mode='max', patience=5, restore_best_weights=True, verbose=0)

        # Set timer
        start = time.time()

        # Train model
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)

        # Stop timer
        duration = time.time() - start

        # Get val_loss and val_iou
        val_loss = history.history["val_loss"][-1]
        val_mean_iou = history.history['val_mean_iou_keras'][-1]
        print(f"val_loss: {val_loss:.6f} | val_mean_iou: {val_mean_iou:.6f} | time: {duration:.3f}s\n")

        # Save the best model params
        if val_mean_iou > best_iou:
            best_iou = val_mean_iou
            best_params = [lr, batch_size, dense_config, val_loss, val_mean_iou]        
            model.save(model_params_file)
        # Update list in top_models_result_file for later checking
        if val_mean_iou > valid_iou:
            top.append([lr, batch_size, dense_config, val_loss, val_mean_iou])

    return best_params, best_iou, top


def test_best_output(X_test, Y_test, best_params, best_iou):
    if best_params is not None:
        print(f"\nBest IoU: {best_iou:.6f} with params: learning_rate: {best_params[0]}, batch_size: {best_params[1]}, dense_config: {best_params[2]}")

        # Test model with the best selected parameters
        test_model = load_model(model_params_file, custom_objects={'mean_iou': mean_iou_keras})
        test_loss, test_iou = test_model.evaluate(X_test, Y_test, verbose=0)
        print(f"Test loss: {test_loss:.6f} | Test mean_iou: {test_iou:.6f}\n")
    else:
        print("No valid model configuration produced a valid IoU. Check for NaNs in training.")


def top_to_list(top, valid_iou):
    if top:
        print(f"Combinations with mean_iou above {valid_iou}:")
        with open(top_models_result_file, 'w') as f:
            for i in top:
                f.write(str(i) + '\n')
                print(i)


# ==================== EXECUTABLE ====================
def main(learning_rates, batch_sizes, dense_layer_config, img_width, img_height, epochs, valid_iou):
    # Load data
    X_train, Y_train, X_test, Y_test, X_val, Y_val, true_size_train, true_size_test, true_size_val, filename_train, filename_test, filename_val = dp.load_data(sources, img_width, img_height)

    count_num_of_combinations(learning_rates, batch_sizes, dense_layer_config)
    output_folder(models_path)
    best_params, best_iou, top = grid_search(learning_rates, batch_sizes, dense_layer_config, img_width, img_height, X_train, Y_train, X_val, Y_val, epochs, valid_iou)
    test_best_output(X_test, Y_test, best_params, best_iou)
    top_to_list(top, valid_iou)


if __name__ == "__main__":
    main(learning_rates, batch_sizes, dense_layer_config, img_width, img_height, epochs, valid_iou)