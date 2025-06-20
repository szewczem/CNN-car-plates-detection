from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bbox_accuracy import intersection_over_union
from data_preparation import rescale_bbox
import time
import progressbar
import random
import cv2


class CNN:
    def __init__(self):
        self.layers = []
        print("Convolutional Neural Network initialized.")

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def print_model_structure(self):
        print("=" * 55)
        print("Model structure:") 
        for i, layer in enumerate(self.layers):
            print(f"{i + 1}. {layer}")
        print("=" * 55)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad_output, learning_rate):
        for layer in self.layers[::-1]:
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output
    
    def batch_generator(self, X, Y, batch_size):
        total_samples = X.shape[0]
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            yield X[i:end_idx], Y[i:end_idx]
    
    def train(self, X, Y, true_size_train, X_val, Y_val, true_size_val, batch_size, loss_fn, epochs, learning_rate):
        print("Training process...", flush=True)
        loss_per_epochs_train = []
        loss_per_epochs_val = []
        accuracy_per_epoch_train = []
        accuracy_per_epoch_val = []
        rescaled_bboxs_predicted_list = []
        rescaled_bboxs_original_list = []

        train_start_time = time.time()              
        
        # Loop through epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
            epoch_start_time = time.time()

            total_loss = 0
            total_samples = X.shape[0]
            Y_predicted_per_epochs = []

            # Initialize progress bar
            num_batches = int(np.ceil(total_samples / batch_size))
            bar_train = progressbar.ProgressBar(maxval=num_batches, widgets=[f'Training ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar_train.start()

            # ========== TRAINING ==========
            # Loop through batchs
            for batch_index, (X_batch, Y_batch) in enumerate(self.batch_generator(X, Y, batch_size)):
                # Forward
                Y_predicted = self.forward(X_batch)
                Y_predicted_per_epochs.append(Y_predicted)                
                # Loss function
                loss = loss_fn.forward(Y_predicted, Y_batch)
                total_loss += loss                
                grad = loss_fn.backward()
                # Backward
                self.backward(grad, learning_rate)
                # Progress bar update
                bar_train.update(batch_index + 1)
            bar_train.finish()             

            # Average loss per epoch
            avg_loss = total_loss / total_samples
            loss_per_epochs_train.append(avg_loss)
            
            # Keep output as shape (batch_size, 4)
            bbox_predicted = np.concatenate(Y_predicted_per_epochs, axis=0)

            # Get resized bbox values
            rescaled_bboxs_predicted = rescale_bbox(bbox_predicted, true_size_train)
            rescaled_bboxs_original = rescale_bbox(Y, true_size_train)            
            if epoch == epochs - 1:
                rescaled_bboxs_predicted_list = rescaled_bboxs_predicted
                rescaled_bboxs_original_list = rescaled_bboxs_original
            
            # Accuracy per epoch
            iou = intersection_over_union(rescaled_bboxs_predicted, rescaled_bboxs_original)
            avg_iou = np.mean(iou)
            accuracy_per_epoch_train.append(avg_iou)
            
            # ========== VALIDATION ==========
            # Initialize progress bar
            total_samples_val = X_val.shape[0]
            num_batches_val = int(np.ceil(total_samples_val / batch_size))
            bar_val = progressbar.ProgressBar(maxval=num_batches_val, widgets=[f'Validation ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar_val.start()

            Y_predicted_val = []
            val_total_loss = 0

            for batch_index, (X_val_batch, Y_val_batch) in enumerate(self.batch_generator(X_val, Y_val, batch_size)):
                pred_val = self.forward(X_val_batch)
                Y_predicted_val.append(pred_val)
                val_loss = loss_fn.forward(pred_val, Y_val_batch)
                val_total_loss += val_loss
                bar_val.update(batch_index + 1)
            bar_val.finish() 

            # Average loss
            val_avg_loss = val_total_loss / X_val.shape[0]
            loss_per_epochs_val.append(val_avg_loss)

            # Keep output as shape (batch_size, 4)
            val_predicted_all = np.concatenate(Y_predicted_val, axis=0)

            # Get resized bbox values
            rescaled_val_pred = rescale_bbox(val_predicted_all, true_size_val)
            rescaled_val_true = rescale_bbox(Y_val, true_size_val)

            # Accuracy
            val_iou = intersection_over_union(rescaled_val_pred, rescaled_val_true)
            val_avg_iou = np.mean(val_iou)
            accuracy_per_epoch_val.append(val_avg_iou)

            # Time per epoch
            epoch_duration = time.time() - epoch_start_time
            
            print(f"time: {epoch_duration:.2f}s | loss: {avg_loss:.6f} | accuracy (IoU): {avg_iou:.6f} | val_loss: {val_avg_loss:.6f} | val_accuracy (IoU): {val_avg_iou:.6f}", flush=True)

        total_train_duration = time.time() - train_start_time
        print(f"\nTraining time: {total_train_duration:.2f} seconds.")
        return loss_per_epochs_train, loss_per_epochs_val, accuracy_per_epoch_train, accuracy_per_epoch_val, rescaled_bboxs_predicted_list, rescaled_bboxs_original_list

    def summary(self, input_shape):
        print("{:<15} {:<25} {:<15}".format("Layer (type)", "Output Shape", "Param #"))
        print("=" * 55)
        
        total_params = 0
        current_shape = (None, *input_shape[1:])
        samples = current_shape[0]

        for layer in self.layers:
            layer_name = layer.__class__.__name__

            if isinstance(layer, Conv):
                num_filters = layer.num_filters
                filter_size = layer.filter_size
                channels = layer.channels
                biases = layer.biases.size

                output_heght = current_shape[1] - filter_size + 1
                output_width = current_shape[2] - filter_size + 1

                output_shape = (samples, output_heght, output_width, num_filters)
                param_count = num_filters * (filter_size * filter_size * channels) + biases

            if isinstance(layer, ReLU):
                output_shape = current_shape
                param_count = 0

            if isinstance(layer, Flatten):
                output_shape = (samples, current_shape[1] * current_shape[2] * current_shape[3])
                param_count = 0
                
            if isinstance(layer, MaxPool):
                filter_size = layer.filter_size

                output_heght = current_shape[1] // filter_size
                output_width = current_shape[2] // filter_size
                channels = current_shape[3]

                output_shape = (samples, output_heght, output_width, channels)
                param_count = 0

            if isinstance(layer, Dense):
                input_size = current_shape[1]
                output_size = layer.output_size

                output_shape = (samples, output_size)
                param_count = input_size * output_size + output_size

            print("{:<15} {:<25} {:<15}".format(layer_name, str(output_shape), str(param_count)))
            total_params += param_count
            current_shape = output_shape
        print("=" * 55)
        print(f"Total params: {total_params}")

    def get_parameters(self):
        # Parameters for all layers
        parameters = {}

        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            prefix = f"{i + 1}_{layer_name}"

            # Check if layer has kernels, biases or weights
            has_kernels = hasattr(layer, 'kernels')
            has_biases = hasattr(layer, 'biases')
            has_weights = hasattr(layer, 'weights')

            if has_kernels and has_biases:
                parameters[f'{prefix}_kernels'] = layer.kernels.copy()
                parameters[f'{prefix}_biases'] = layer.biases.copy()

            if has_weights and has_biases:
                parameters[f'{prefix}_weights'] = layer.weights.copy()
                parameters[f'{prefix}_biases'] = layer.biases.copy()
        
        return parameters
    
    def save_parameters(self, filename="custom_model_parameters.npz"):
        params = self.get_parameters()
        np.savez(filename, **params)

    @staticmethod
    def load_parameters(filename="custom_model_parameters.npz"):
        loaded = np.load(filename)
        return {key: loaded[key] for key in loaded.files}
    
    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            prefix = f"{i + 1}_{layer_name}"

            # Keys in dict
            k_key = f"{prefix}_kernels"
            w_key = f"{prefix}_weights"
            b_key = f"{prefix}_biases"

            if hasattr(layer, 'kernels') and k_key in params:
                layer.kernels = params[k_key]
            if hasattr(layer, 'weights') and w_key in params:
                layer.weights = params[w_key]
            if hasattr(layer, 'biases') and b_key in params:
                layer.biases = params[b_key]

    @staticmethod
    def plot_loss_for_learning(loss_per_epochs_train, loss_per_epochs_val, learning_rate):
        plt.plot(loss_per_epochs_train, label='Train Loss')
        plt.plot(loss_per_epochs_val, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train vs Validation Loss (learning_rate = {learning_rate})')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_accuracy_for_learning(accuracy_per_epoch_train, accuracy_per_epoch_val, learning_rate):
        plt.plot(accuracy_per_epoch_train, label='Train IoU')
        plt.plot(accuracy_per_epoch_val, label='Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Intersection over Union')
        plt.title(f'Train vs Validation Accuracy (IoU) (learning_rate = {learning_rate})')
        plt.legend()
        plt.grid(True)
        plt.show()
