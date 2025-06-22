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
import os


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
    
    def train(self, X, Y, true_size_train, X_val, Y_val, true_size_val, batch_size, loss_fn, epochs, learning_rate, min_delta, patience, filename):
        '''
        Training process for model. Train on training set, validate on val set, early stop if val_avg_loss doesn't improve for indicated number of epochs (patience).

        Parameters:
            X -- array with training values
            Y -- array with true bbox values for training set
            true_size_train -- array with original img size for training set
            X_val -- array with validation values
            Y_val -- array with true bbox values for validation set
            true_size_val -- array with original img size for validation set
            batch_size -- batch size of input (equal for training and validation)
            loss_fn -- provided loss function
            epochs -- number of epochs
            learning_rate -- learning rate for backward propagation update
            min_delta -- min value for val_iou that can be takean as improvement
            patience -- number of epochs of no improvments in validation loss
            filename -- filename for saving best params, extension .npz
        '''

        print("Training process...", flush=True)
        loss_per_epochs_train = []
        loss_per_epochs_val = []
        accuracy_per_epoch_train = []
        accuracy_per_epoch_val = []
        rescaled_bboxs_pred_train = []
        rescaled_bboxs_true_train = []
        rescaled_bboxs_pred_val = []
        rescaled_bboxs_true_val = []

        train_start_time = time.time()           

        # Early stopping
        best_val_iou = -1
        noimprovement = 0
        
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

            Y_predicted_per_epochs_val = []
            val_total_loss = 0

            for batch_index, (X_val_batch, Y_val_batch) in enumerate(self.batch_generator(X_val, Y_val, batch_size)):
                Y_predicted_val = self.forward(X_val_batch)
                Y_predicted_per_epochs_val.append(Y_predicted_val)
                val_loss = loss_fn.forward(Y_predicted_val, Y_val_batch)
                val_total_loss += val_loss
                bar_val.update(batch_index + 1)
            bar_val.finish() 

            # Average loss
            val_avg_loss = val_total_loss / X_val.shape[0]
            loss_per_epochs_val.append(val_avg_loss)
            
            # Keep output as shape (batch_size, 4)
            bbox_predicted_val = np.concatenate(Y_predicted_per_epochs_val, axis=0)

            # Get resized bbox values
            rescaled_val_pred = rescale_bbox(bbox_predicted_val, true_size_val)
            rescaled_val_true = rescale_bbox(Y_val, true_size_val)

            # Accuracy
            val_iou = intersection_over_union(rescaled_val_pred, rescaled_val_true)
            val_avg_iou = np.mean(val_iou)
            accuracy_per_epoch_val.append(val_avg_iou)

            # Time per epoch
            epoch_duration = time.time() - epoch_start_time

            # ========== EARLY STOPPING ==========
            if val_avg_iou > best_val_iou + min_delta:
                best_val_iou = val_avg_iou
                noimprovement = 0
                self.save_parameters(filename)

                # bbox predictions
                rescaled_bboxs_pred_train = rescaled_bboxs_predicted
                rescaled_bboxs_true_train = rescaled_bboxs_original
                rescaled_bboxs_pred_val = rescaled_val_pred
                rescaled_bboxs_true_val = rescaled_val_true
            else:
                noimprovement += 1

            if noimprovement >= patience:
                print(f"Early stopping triggered after {epoch + 1}/{epochs}.", flush = True)
                self.set_parameters(filename)
                break
            
            # Summary of epoch
            print(f"time: {epoch_duration:.2f}s | loss: {avg_loss:.6f} | accuracy (IoU): {avg_iou:.6f} | val_loss: {val_avg_loss:.6f} | val_accuracy (IoU): {val_avg_iou:.6f}", flush=True)
        
        total_train_duration = time.time() - train_start_time
        print(f"\nTraining time: {total_train_duration:.2f} seconds.")
        return loss_per_epochs_train, loss_per_epochs_val, accuracy_per_epoch_train, accuracy_per_epoch_val, rescaled_bboxs_pred_train, rescaled_bboxs_true_train, rescaled_bboxs_pred_val, rescaled_bboxs_true_val
    
    def test(self, X_test, Y_test, true_size_test, batch_size, loss_fn, filename):
        print("Testing the model...", flush=True)

        self.set_parameters(filename)

        test_start_time = time.time()    

        # Initialize progress bar
        total_samples_test = X_test.shape[0]
        num_batches_test = int(np.ceil(total_samples_test / batch_size))
        bar_test = progressbar.ProgressBar(maxval=num_batches_test, widgets=[f'Testing ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar_test.start()

        Y_predicted_test_all = []
        test_total_loss = 0

        for batch_index, (X_test_batch, Y_test_batch) in enumerate(self.batch_generator(X_test, Y_test, batch_size)):
            Y_predicted_test = self.forward(X_test_batch)
            Y_predicted_test_all.append(Y_predicted_test)
            test_loss = loss_fn.forward(Y_predicted_test, Y_test_batch)
            test_total_loss += test_loss
            bar_test.update(batch_index + 1)
        bar_test.finish() 

        # Average loss
        test_total_loss = test_total_loss / X_test.shape[0]
        
        # Keep output as shape (batch_size, 4)
        bbox_predicted_test = np.concatenate(Y_predicted_test_all, axis=0)

        # Get resized bbox values
        rescaled_bboxs_test_pred = rescale_bbox(bbox_predicted_test, true_size_test)
        rescaled_bboxs_test_true = rescale_bbox(Y_test, true_size_test)

        # Accuracy
        test_iou = intersection_over_union(rescaled_bboxs_test_pred, rescaled_bboxs_test_true)
        test_avg_iou = np.mean(test_iou)

        # Testing time
        test_duration = time.time() - test_start_time

        print(f"\nTesting time: {test_duration:.2f} seconds | test_loss: {test_total_loss:.6f} | test_accuracy (IoU): {test_avg_iou:.6f}", flush=True)
        return test_total_loss, test_avg_iou, rescaled_bboxs_test_pred, rescaled_bboxs_test_true
    
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

            # For Conv
            if has_kernels and has_biases:
                parameters[f'{prefix}_kernels'] = layer.kernels.copy()
                parameters[f'{prefix}_biases'] = layer.biases.copy()

            # For Dense
            if has_weights and has_biases:
                parameters[f'{prefix}_weights'] = layer.weights.copy()
                parameters[f'{prefix}_biases'] = layer.biases.copy()     
                   
        return parameters
    
    def save_parameters(self, filename):
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.mkdir(folder)
        params = self.get_parameters()
        np.savez(filename, **params)

    @staticmethod
    def load_parameters(filename):
        params = np.load(filename)
        return {key: params[key] for key in params.files}
    
    def set_parameters(self, filename):
        params = self.load_parameters(filename)
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
        plt.title(f'Model Loss (lr = {learning_rate})')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_accuracy_for_learning(accuracy_per_epoch_train, accuracy_per_epoch_val, learning_rate):
        plt.plot(accuracy_per_epoch_train, label='Train IoU')
        plt.plot(accuracy_per_epoch_val, label='Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Intersection over Union')
        plt.title(f'Model Accuracy (IoU) (lr = {learning_rate})')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_predicted_vs_true_bboxes(Y_true_rescaled, Y_pred_rescaled, filenames, source_dirs, set):
        """
        Plot 12 random images with true and predicted bounding boxes.

        Parameters:
            Y_true_rescaled -- array of shape (N, 4)
            Y_pred_rescaled -- array of shape (N, 4)
            filenames -- list of image filenames
            source_dirs -- list of folders to search for the images
            set -- str added to the end of plot title
        """
        # 12 or less if not enough provided
        num_samples = min(12, len(filenames))
        # Select random num_samples idx from all avaible
        indices = np.random.choice(len(filenames), num_samples, replace=False)

        fig, axes = plt.subplots(4, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            fname = filenames[idx]

            # Find full path
            img_path = None
            for folder in source_dirs:
                try_path = os.path.join(folder, fname)
                if os.path.exists(try_path):
                    img_path = try_path
                    break

            # Load and prepare image
            # Load and prepare image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_copy = img.copy()

            # True bbox - green
            xtl, ytl, xbr, ybr = np.array(Y_true_rescaled[idx]).astype(int)
            cv2.rectangle(img_copy, (xtl, ytl), (xbr, ybr), (0, 255, 0), 8)

            # Predicted bbox - red
            xtl, ytl, xbr, ybr = np.array(Y_pred_rescaled[idx]).astype(int)
            cv2.rectangle(img_copy, (xtl, ytl), (xbr, ybr), (255, 0, 0), 8)

            axes[i].imshow(img_copy)
            axes[i].set_title(fname)
            axes[i].axis("off")

            for ax in axes[num_samples:]:
                ax.axis('off')

        fig.suptitle(f"Original (green) vs Predicted (red) Bounding Boxes for {set}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()