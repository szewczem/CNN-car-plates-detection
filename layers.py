import numpy as np
import data_preparation as dp


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self):
        pass


'''
1. Input: image array of shape (400,640,1)
2. First Convolutional layer: filter (3,3), number of kernels = 8, padding = 0
    output_height = input_height - filter + 1
    output_width = input_width - filter + 1
    kernels (8,3,3,1)
    output (398,638,n), where n = number of filters (kernels) => (398,638,8)
3. ReLu Activation f(x) = max(0,x)
4. Pooling (MaxPooling), stride = 2, size 2x2, output => (199,319,8)
5. Repeat 2-4 (second Convolution), input (199,319,8), number of kernels = 16 => (197,317,16), Relu => (197,317,16), Pooling => (98,158,16)
6. Flattening 98*158*16 => (1, 247744)
7. Fully Connected (hiden and dense) => (1,4)
'''

class Conv:
    def __init__(self, num_filters, filter_size, input_channels=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.kernels = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.1    # initialization of n filters (kernels), array filter_sizexfilter_size with random values from normal distribution scales down by /10, (num_filters, height, width, input_channels), for grayscale the input_channels = 1
        print('Convolutional layer initialized.')

    # return a one piece (slice) of array indicated by filter at a time, h - row idx, w - column idx
    def image_region(self, input):
        self.input = input
        batch_size, height, width, channels = input.shape
        f = self.filter_size

        # height and width of output array
        out_height = height - f + 1
        out_width = width - f + 1

        for n in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    patch = input[n, h:(h + f), w:(w + f), :]    # patch shape (f, f, num_filters)
                    yield patch, n, h, w    # return and pause

    def forward(self, input):
        self.input = input    # input array of input, shape (inp_height, inp_width, 1)
        batch_size, height, width, channels = input.shape
        num_filters = self.num_filters
        f = self.filter_size

        # height and width of output array
        out_height = height - f + 1
        out_width = width - f + 1

        # shape of output array
        conv_out = np.zeros((batch_size, out_height, out_width, num_filters))

        for patch, n, h, w in self.image_region(input):
            conv_out[n, h, w, :] = np.sum(patch * self.kernels, axis = (1,2,3))

        print(f'Image patch shape: {patch.shape}')
        print(f'Filters shape: {self.kernels.shape}')
        print(f'Image patch * kernel shape: {(patch * self.kernels).shape}')
        print(f'Output shape: {conv_out.shape}')
        return conv_out

    def backward(self, grad_output, learning_rate):
        input = self.input
        batch_size, height, width, channels = input.shape
        f = self.filter_size
        num_filters = self.num_filters

        # Output shape
        out_height = height - f + 1
        out_width = width - f + 1
        
        grad_conv = np.zeros(input.shape)
        grad_kernels = np.zeros_like(self.kernels)

        for patch, n, h, w in self.image_region(input):
            for k in range(num_filters):
                grad_kernels[k] += patch * grad_output[n, h, w, k]    # patch shape (f, f, channels), kernels change
                grad_conv[n, h:(h + f), w:(w + f), :] += self.kernels[k] * grad_output[n, h, w, k]    # gradiend change

        # filter params update
        self.kernels -= learning_rate * grad_kernels
        return grad_conv


class ReLU:
    def __init__(self):
        print("ReLU activation initialized.")

    def forward(self, input_data):
        # if x <= 0 then 0
        self.input_data = input_data 
        relu_out = np.maximum(0, input_data)
        return relu_out

    def backward(self, grad_output):
        # when input > 0, then gradient is 1
        grad = self.input_data > 0
        grad_relu = grad_output * grad
        return grad_relu


class MaxPool:
    def __init__(self, filter_size):
        self.filter_size = filter_size
        print('Max Pooling layer initialized.')

    # return a one piece (slice) of array indicated by filter at a time, j - row idx, k - column idx
    def image_region(self, input):        
        batch_size, height, width, num_filters = input.shape
        f = self.filter_size

        # height and width of output array
        # for filter of size nxn and stride n, the output dimensions will be n times smaller in each dimension (height and width)
        out_height = height // f   
        out_width = width // f

        for n in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    patch = input[n, (h * f):(h * f + f), (w * f):(w * f + f), :]    # (n, h*f:h*f+f, w*f:w*f+f, :) -> (f,f,num_filters)
                    yield patch, n, h, w    # return and pause

    def forward(self, input):
        self.input = input
        batch_size, height, width, num_filters = input.shape
        f = self.filter_size

        # height and width of output array
        out_height = height // f   
        out_width = width // f

        # shape of output array
        maxpool_out = np.zeros((batch_size, out_height, out_width, num_filters))

        for patch, n, h, w in self.image_region(input):
            maxpool_out[n, h, w, :] = np.amax(patch, axis = (0,1))

        print(f'Image patch shape: {patch.shape}')
        print(f'Output shape: {maxpool_out.shape}')
        return maxpool_out

    def backward(self, grad_output):
        # gradient only to the position of the max value (contribute to the output), all other positions set to 0
        input = self.input
        batch_size, height, width, num_filters = input.shape
        f = self.filter_size        

        # height and width of output array
        out_height = height // f   
        out_width = width // f

        grad_maxpool = np.zeros(input.shape)

        for n in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    patch = input[n, (h * f):(h * f + f), (w * f):(w * f + f), :]    # shape (f, f, :)
                    max_vals = np.amax(patch, axis=(0, 1), keepdims=True)    # keep the max value from the patch
                    mask = (patch == max_vals)    # in patch the max value gets 1 (others get 0)

                    grad = mask * grad_output[n, h, w, :]

                    grad_maxpool[n, (h * f):(h * f + f), (w * f):(w * f + f), :] += grad
        return grad_maxpool    # shape (n, height, width, num_filters)
                

class Flatten:
    def __init__(self):
        print('Flattening layer initialized.')

    def forward(self, input):
        # flat_out = []
        # for row in image:
        #     for col in row:
        #         for val in col:
        #             flat_out.append(val)
        # flat_out = np.array([flat_out])    # provide the 1D shape (1,n), where n - features

        # creating one dimensional list 
        self.input_shape = input.shape
        batch_size = input.shape[0]
        flat_out = input.flatten().reshape(batch_size, -1)    # batch_size for number of examples (rows in array), -1 for the number of last element in total features (columns in array)
        print(f'Output shape: {flat_out.shape}')
        return flat_out
        
    def backward(self, grad_output):
        # grad_output.shape (batch_size, features), (batch_size, features = heightxwidthxdim) reshape it to forward input (batch_size, height, width, dim)
        grad_flat = grad_output.reshape(self.input_shape)
        return grad_flat    # (batch_size, height, width, dim)


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        print(f'FullyConnected initialized: input_size={input_size}, output_size={output_size}')

    def forward(self, input):
        self.input = input

        # output = input x weights + biases
        fc_out = np.dot(input, self.weights) + self.bias
        print(f'FC output shape: {fc_out.shape}')
        return fc_out

    def backward(self, grad, learning_rate):
        '''
        input (1, input_size) => (1, 247744)
        weights (input_size, 4) => (247744, 4)
        bias (1, output_size) => (1, 4)
        output = input x weights + biases => output (1, 4)
        grad_output (1, 4)

        l - loss, w - weights, b - bias

        l = 1/n * ∑(y_pred - y)^2
        y_pred = w * x + b

        1. dl/dw = dy_pred/dw * dl/dy_pred
            dy_ped/dw = x
            dl/dy_pred = 1/n * ∑(y_pred^2 - 2*y_pred*y + y^2)   =>   dl/dy_pred = 1/n * ∑(2*y_pred - 2*y)   => 2/n * ∑(y_pred - y) - ∑grad
            
            dl/dw = x * 2/n * ∑(y_pred - y)   =>   x * ∑grad

            dl/dw = x * ∑grad   ->   x.shape (1,247744), grad.shape (1,4), grad_output.shape (247744,4)   =>   dl/dw = .dot(x.T,gard)

        2. dl/db = dl/dy_pred * dy_pred/db
            dl/dy_pred = 2/n * ∑(y_pred - y)
            dy_pred/db = 1

            dl/db = 2/n * ∑(y_pred - y) * 1   =>   2/n * ∑(y_pred - y)   =>   ∑grad

            dl/db = np.sum(grad)   ->   grad.shape(1, 4)
        '''
        # gradients
        # dL/dW
        grad_weights = np.dot(self.input.T, grad)    # (input_size, output_size) => (247744, 4)
        # dL/db
        grad_bias = np.sum(grad, axis=0, keepdims=True)    # (1, 4), summing all rows
        # dL/dX
        grad_fc = np.dot(grad, self.weights.T)

        # weight and bias update
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_fc

'''
Forwardpropagation:
Conv1 -> ReLU1 -> MaxPool1 -> Conv2 -> ReLU2 -> MaxPool2 -> Flatten -> FC -> MSELoss
X -> conv_out1 -> relu_out1 -> max_pool_out1 -> conv_out2 -> relu_out2 -> max_pool_out2 -> flat_out -> fc_out -> Y_pred

Backpropagation:
MSELoss -> FC -> Flatten -> MaxPool2 -> ReLU2 -> Conv2 -> MaxPool1 -> ReLU1 -> Conv1


Flatten:
    input (1, 247744)
FullyConnected:
    input (1, 247744) - flat_out
    w.shape (247744, 4) - 4 bbox outputs
    b.shape (1, 4) - 4 bbox outputs
    output (1, 4)
'''

class MSELoss:
    def __init__(self):
        print("Mean Squared Error loss initialized.")

    def forward(self, y_original, y_predicted):
        self.y_original = y_original
        self.y_predicted = y_predicted

        # loss = 1/n * ∑(y_original - y_predicted)^2, where n - total number of elements (batch * bbox)
        loss = np.mean((y_predicted - y_original) ** 2)
        return loss

    def backward(self):
        # gradient of the loss, batch size = 2^n
        # dl/dy_pred = 2/n * (y_pred- y), where n - total number of elements (batch * bbox)
        n = self.y_original.size
        grad = (2 / n) * (self.y_predicted - self.y_original)
        return grad
