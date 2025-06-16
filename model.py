from layers import Conv, ReLU, MaxPool, Flatten, Dense, MSELoss
import pandas as pd


class CNN:
    def __init__(self):
        self.layers = []
        print("Convolutional Neural Network initialized.")

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad_output):
        for layer in self.layers[::-1]:
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def batch_generator(self, X, Y, batch_size):
        total_samples = X.shape[0]
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            yield X[i:end_idx], Y[i:end_idx]
    
    def train(self, X, Y, batch_size, loss_fn, epochs, learning_rate):
        for epoch in epochs:
            total_loss = 0
            for X_batch, Y_batch in self.batch_generator(X, Y, batch_size):
                Y_predicted = self.forward(X_batch)

                loss = loss_fn.forward(Y_predicted, Y_batch)
                total_loss += loss

                grad = loss_fn.backward()
                self.backward(grad)

            avg_loss = total_loss / (X.shape[0] // batch_size)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def print_model_structure(self):
        print("="*50)
        print("Model structure:")
        for i, layer in enumerate(self.layers):
            print(f"{i + 1}. {layer}")
        print("="*50)

    def summary(self):
        data = {'Layer': [], 'Output shape': [], 'Parameters': []}

        for layer in self.layers:
            data.appeend()
        df = df.DataFrame(data)
        print(df)