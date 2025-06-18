from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bbox_accuracy import intersection_over_union
from data_preparation import rescale_bbox


class CNN:
    def __init__(self):
        self.layers = []
        print("Convolutional Neural Network initialized.")

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def print_model_structure(self):
        print("="*50)
        print("Model structure:")
        for i, layer in enumerate(self.layers):
            print(f"{i + 1}. {layer}")
        print("="*50)

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
    
    def train(self, X, Y, original_size, batch_size, loss_fn, epochs, learning_rate):
        print("Training process...")
        loss_per_epochs = []
        accuracy_per_epoch = []
        rescaled_bboxs_predicted_list = []
        rescaled_bboxs_original_list = []
        
        # Loop through epochs
        for epoch in range(epochs):
            total_loss = 0
            total_samples = X.shape[0]
            y_predicted_per_epochs = []

            # Loop through batchs
            for X_batch, Y_batch in self.batch_generator(X, Y, batch_size):
                # Forward
                Y_predicted = self.forward(X_batch)
                y_predicted_per_epochs.append(Y_predicted)                
                # Loss function
                loss = loss_fn.forward(Y_predicted, Y_batch)
                total_loss += loss                
                grad = loss_fn.backward()
                # Backward
                self.backward(grad, learning_rate)

            # Loss per epoch
            avg_loss = total_loss / total_samples
            loss_per_epochs.append(avg_loss)
            
            # Keep output as shape (batch_size, 4)
            bbox_predicted = np.concatenate(y_predicted_per_epochs, axis=0)

            # Get resized bbox values
            rescaled_bboxs_predicted = rescale_bbox(bbox_predicted, original_size)
            rescaled_bboxs_original = rescale_bbox(Y, original_size)
            
            if epoch == epochs - 1:
                rescaled_bboxs_predicted_list = rescaled_bboxs_predicted
                rescaled_bboxs_original_list = rescaled_bboxs_original
            
            # Accuracy per epoch
            iou = intersection_over_union(rescaled_bboxs_predicted, rescaled_bboxs_original)
            avg_accuracy = sum(iou) / len(iou)
            accuracy_per_epoch.append(avg_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Mean IoU: {avg_accuracy:.6f}")
        return loss_per_epochs, avg_accuracy, rescaled_bboxs_predicted_list, rescaled_bboxs_original_list

    def summary(self):
        pass

    def plot_loss_for_learning(self, loss_per_epochs):
        plt.plot(loss_per_epochs)
        plt.ylabel('Loss function')
        plt.xlabel('Epoch')
        plt.title(f"Training Loss over Epochs, learning_rate = ")
        plt.show()