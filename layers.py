import numpy as np
from scipy import signal
import data_preparation as dp


class Layer:
    def __init__(self):
        print('Layer initialized.')
    
    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

'''
1. Input: image array of shape (400,640,1)
2. First Convolutional layer: filter (3,3), number of kernels = 8, padding = 0, stride = 1
    output_height = input_height - filter + 1
    output_width = input_width - filter + 1
    kernels (8,3,3,1)
    output (398,638,n), where n = number of filters/kernels => (398,638,8)
3. ReLu Activation f(x) = max(0,x)
4. Pooling (MaxPooling), stride = 2, size 2x2, output => (199,319,8)
5. Repeat 2-4 (second Convolution), input (199,319,8), number of kernels = 16 => (197,317,16), Relu => (197,317,16), Pooling => (98,158,16)
6. Flattening 98*158*16 => (1, 247744)
7. Fully Connected (hiden and dense) => (1,4)
'''


class Conv(Layer):
    '''
    Convolutional layer that applies filters (kernels) to input data to detect features (edges, patterns).

    Arguments: 
    num_filters -- int, number of filters (kernels)
    filter_size -- int, height and width for each filter (kernel), square shape
    input_channels -- int, number of channels in the input data (1 for grayscale, 3 for RGB - for first covolution only).

    Forward:
        Arguments:
            input -- array of shape (batch_size, height, width, channels)
            For the second or later convolutioal layers, channels number is equal to number of filters (kernels) from previous convolution.
        Returns:
            conv_out -- array of shape (batch_size, height, width, num_filters), where num_filters is equal to provided number of kernels, the result of convolving filters over the input

    Backward:
        Arguments:
            grad_output -- array of shape (batch_size, height, width, num_filters), the gradient propagated from the previous layer (ReLU)
            learning_rate -- float, the step size used to update model parameters (kernel weights)
        Returns:
            grad_conv -- array of shape (batch_size, height, width, input_channels), gradient of the loss with respect to the input, the shape is corresponding to forward input            
    '''

    def __init__(self, num_filters, filter_size, input_channels=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels        
        # initialization of n filters (kernels) with random values from normal distribution scales down by 10
        self.kernels = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.1
        self.biases = np.zeros(num_filters)    # biases start at 0
        print('Convolutional layer initialized.')

    # return a one piece (slice) of input array indicated by filter at a time, h - row idx, w - column idx
    def image_region(self, input):
        self.input = input
        batch_size, height, width, channels = input.shape
        f = self.filter_size

        # height and width of output array, last common row and column is the end of output array
        out_height = height - f + 1
        out_width = width - f + 1

        for n in range(batch_size):    # loop through batch samples
            for h in range(out_height):    # loop through its rows (height)
                for w in range(out_width):    # loop through its columns (width)
                    patch = input[n, h:(h + f), w:(w + f), :]    # patch shape (f, f, channels) for image n in batch_size
                    yield patch, n, h, w    # return and pause

    # return the output array (feature map) of shape (batch_size, out_height, out_width, num_filters) where all elements from image_region() are multipled by kernel and summed
    def forward(self, input):
        self.input = input  
        batch_size, height, width, channels = input.shape
        num_filters = self.num_filters
        f = self.filter_size
        b = self.biases

        # height and width of output array
        out_height = height - f + 1
        out_width = width - f + 1

        # shape of output array filled by 0
        conv_out = np.zeros((batch_size, out_height, out_width, num_filters))

        # output array filled by sum of the multiplication between each input patch and each filter (kernel)
        for patch, n, h, w in self.image_region(input):
            conv_out[n, h, w, :] = np.sum(patch * self.kernels, axis = (1,2,3)) + b    # 4D array filled by 1D output (one pixel/feature in image for every filters)

        print(f'Image patch shape: {patch.shape}')
        print(f'Filters shape: {self.kernels.shape}')
        print(f'Biases shape: {b.shape}')
        print(f'Image patch * kernel shape: {(patch * self.kernels).shape}')
        print(f'Output shape: {conv_out.shape}')
        return conv_out
    
    # kernels rotation function, rotate over rows and columns, channel stay intact
    def rotate180(self, kernel):
        kernel_180 = kernel[::-1, ::-1, :]
        return kernel_180

    def backward(self, grad_output, learning_rate):
        input = self.input    # from forward
        batch_size, height, width, channels = input.shape
        f = self.filter_size
        num_filters = self.num_filters
        
        # arrays shape for kernels and conv gradients 
        grad_kernels = np.zeros(self.kernels.shape)    # array of ashape (num_filters, filter_size, filter_size, input_channels)
        grad_conv = np.zeros(input.shape)    # array of shape (batch_size, out_height, out_width, num_filters)
        grad_biases = np.zeros(self.biases.shape)      

        '''
        Gradient for kernels:
            dl/dki = ?
            dk -- gradient of the cost with respect of the weights of the kernel
            z -- values from grad_output
            A -- imput array (from forward)
            a -- values from input array (from forward)

            dl/dk = dz/dk * dl/dz   ->   dz/dk = const. = a

            value for each element in output array of convolution:
            z1 = a1*k1 + a2*k2 + ... + an*kn + b           
            z2 = a1*k1 + a2*k2 + ... + an*kn + b
            ...
            zn = a1*k1 + a2*k2 + ... + an*kn + b

            dl/dk = ∑(dz/dk * dl/dz), for the above equations:
            dl/dk1 = a1 * dl/dz1 + a2 * dl/dz2 + ... + an * dl/dzn
            ...
            dl/dkn = a1 * dl/dz1 + a2 * dl/dz2 + ... + an * dl/dzn

            dl/dk = [a] * [dl/dz]   =>   dl/dk = conv(A, dl/dz)   =>   grad_kernels = conv(input, grad_output)

        Gradient for input:
            dl/dai = ?

            dl/da = ∑(dz/da * dl/dz)

            value for each element in output array of convolution:
            z1 = a1*k1 + a2*k2 + a3*k3 + ... + an*km + b,
            z2 = a2*k1 + a3*k2 + a4*k3 +... + an*km + b
            z3 = a3*k1 + a4*k2 + a5*k3 +... + an*km + b
            ...
            zn = an*k1 + an-1*k2 + ... + a1*km + b

            dz/da (for the above equations):
            dl/da1 = k1 * dl/dz1
            dl/da2 = k2 * dl/dz1 + k1 * dl/dz2
            ...
            dl/dan = kn * dl/dz1 + ... + k2 * dl/dzn-1 + k1 * dl/dzn

            dl/da = cov(dl/dz, 180deg*kernel)   ->   dl/dz is padded as its shape is smaller than dl/da

        Gradient for bias:
            dl/db = ?

            dl/db = ∑(dl/dz * dz/db)   ->   dz/db = const. = 1

            value for each element in output array of convolution:
            z1 = a1*k1 + a2*k2 + ... + an*kn + b           
            z2 = a1*k1 + a2*k2 + ... + an*kn + b
            ...
            zn = a1*k1 + a2*k2 + ... + an*kn + b

            dl/db = ∑(dl/dz)
        '''

        # gradient for kernels: loop through patch, where n is the idx of image, h - row idx, w - column idx, update grad of every filter (kernel), dl/dk = [a] * [dl/dz]
        for patch, n, h, w in self.image_region(input):
            for k in range(num_filters):
                grad_kernels[k] += patch * grad_output[n, h, w, k]    # patch shape (f, f, channels), kernels change                

        # gradient for input: for grad_output (image) n from previous layer and all its channels, filter k and all its channels do full convolution, dl/da = cov(dl/dz, 180deg*kernel)
        for n in range(batch_size):
            for k in range(num_filters):          
                rotated_kernel = self.rotate180(self.kernels[k])    # kernel k rotated by 180deg
                grad_output_per_kernel = grad_output[n, :, :, k]    # gradient loss with respect to output filter, shape (h, w)

                for c in range(channels):                    
                    for h in range(grad_output_per_kernel.shape[0]):
                        for w in range(grad_output_per_kernel.shape[1]):
                            grad_conv[n, h:(h + f), w:(w + f), c] += grad_output_per_kernel[h, w] * rotated_kernel[:, :, c]
                # grad_conv[n, :, :, c] += signal.convolve2d(grad_output[n, :, :, k], rotated_kernel[:, :, c], mode="full")

                # gradients for biases, dl/db = ∑(dl/dz)
                grad_biases[k] += np.sum(grad_output_per_kernel)              

        # filter params update, k = k - learning_rate * dl/dk, b = b - learning_rate 
        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases
        return grad_conv


class ReLU:
    '''
    ReLU activation layer that applies 0 for all values lower or equal 0 (positive keeps as-is).

    Forward:
        Arguments:
            input -- array of shape (batch_size, height, width, channels)
        Returns:
            relu_out -- array of shape (batch_size, height, width, channels), where are values are equal or greater than 0

    Backward:
        Arguments:
            grad_output -- array of shape (batch_size, height, width, channels), the gradient propagated from the previous layer (MaxPool)
        Returns:
            grad_relu -- array of shape (batch_size, height, width, channels)       
    '''

    def __init__(self):
        print("ReLU activation initialized.")

    def forward(self, input):
        # f(x) = max(0,x), if input <= 0 then 0
        self.input = input 
        relu_out = np.maximum(0, input)
        return relu_out

    def backward(self, grad_output):
        # when input > 0, then gradient is 1
        # grad_relu.shape == grad_output.shape == grad.shape, relu doesn't change the shape of array
        grad = self.input > 0    # array of true and false
        grad_relu = grad_output * grad    
        return grad_relu


class MaxPool:
    '''
    MaxPool layer reduce the image size by keeping only the biggest value from filtered region.

    Arguments: 
    filter_size -- int, height and width for each filter, square shape

    Forward:
        Arguments:
            input -- array of shape (batch_size, height, width, channels)
        Returns:
            maxpool_out -- array of shape (batch_size, height/filter_size, width/filter_size, channels)

    Backward:
        Arguments:
            grad_output -- array of shape (batch_size, height, width, channels), the gradient propagated from the previous layer (Conv or Flatten)
        Returns:
            grad_maxpool -- array of shape (batch_size, height*filter_size, width*filter_size, channels)        
    '''

    def __init__(self, filter_size):
        self.filter_size = filter_size
        print('Max Pooling layer initialized.')

    # return a one piece (slice) of array indicated by filter at a time, j - row idx, k - column idx
    def image_region(self, input):        
        batch_size, height, width, channels = input.shape
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

        # fill maxpool_out with max value in the patch for each channel
        for patch, n, h, w in self.image_region(input):
            maxpool_out[n, h, w, :] = np.amax(patch, axis = (0,1))    # max value in patch for each channel

        print(f'Image patch shape: {patch.shape}')
        print(f'Output shape: {maxpool_out.shape}')
        return maxpool_out

    def backward(self, grad_output):
        '''
        gradient change has input only for max values in patch, the rest values is equal to 0

        grad_maxpool.shape == input.shape (from forward) 

        dl/dc = dl/dz if c is the max element, 0 otherwise
        '''

        input = self.input
        batch_size, height, width, num_filters = input.shape
        f = self.filter_size        

        # height and width of output array
        out_height = height // f   
        out_width = width // f

        # creating output array filled by 0
        grad_maxpool = np.zeros(input.shape)

        # update grad_maxpool, values that was max in the input filtered patch gets gradient from grad_output
        for patch, n, h, w in self.image_region(input):
            max_vals = np.amax(patch, axis=(0, 1), keepdims=True)    # keep the max value from the patch
            mask = (patch == max_vals)    # if value in patch is max then True, array of shape (h, w, channels)
            grad = mask * grad_output[n, h, w, :]
            grad_maxpool[n, (h * f):(h * f + f), (w * f):(w * f + f), :] += grad
        return grad_maxpool   

        # for n in range(batch_size):
        #     for h in range(out_height):
        #         for w in range(out_width):
        #             patch = input[n, (h * f):(h * f + f), (w * f):(w * f + f), :]    # shape (f, f, :)
        #             max_vals = np.amax(patch, axis=(0, 1), keepdims=True)    # keep the max value from the patch
        #             mask = (patch == max_vals)    # in patch the max value gets 1 (others get 0)
        #             grad = mask * grad_output[n, h, w, :]
        #             grad_maxpool[n, (h * f):(h * f + f), (w * f):(w * f + f), :] += grad
        # return grad_maxpool    # shape (n, height, width, channels)
                

class Flatten:
    '''
    Flatten layer turns input into a long 1D list of numbers.

    Forward:
        Arguments:
            input -- array of shape (batch_size, height, width, channels)
        Returns:
            height * width * channels = n
            flat_out -- array of shape (batch_size, height * width * channels,) -> (batch_size, n)

    Backward:
        Arguments:
            grad_output -- array of shape (batch_size, n), the gradient propagated from the previous layer (FC)
        Returns:
            grad_flat -- array of shape (batch_size, height, width, channels)   
    '''

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
        # convert input array into 1D list of numbers (flatten()), reshape 1D array to 2D array (barch_size, n)
        flat_out = input.flatten().reshape(batch_size, -1)
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
            dl/dy_pred = 1/n * ∑(y_pred^2 - 2*y_pred*y + y^2)   =>   dl/dy_pred = 1/n * ∑(2*y_pred - 2*y)   =>   2/n * ∑(y_pred - y) - ∑grad
            
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
X -> conv_out1 -> relu_out1 -> maxpool_out1 -> conv_out2 -> relu_out2 -> maxpool_out2 -> flat_out -> fc_out -> Y_pred

Backpropagation:
MSELoss -> FC -> Flatten -> MaxPool2 -> ReLU2 -> Conv2 -> MaxPool1 -> ReLU1 -> Conv1
grad -> grad_fc -> grad_flat -> grad_maxpool2 -> grad_relu2 -> grad_conv2 -> grad_maxpool -> grad_relu -> grad_conv
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