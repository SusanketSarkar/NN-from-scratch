import numpy as np
from activation import ReLU, Sigmoid

class Dense:
    def __init__(self, input_size, output_size, activation):
        # He initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, x):
        self.Z = np.dot(self.weights, x) + self.bias
        self.A = self.activation.function(self.Z)
        return self.A
    
    def backward(self, dA, A_p):
        m = A_p.shape[1]  # batch size
        dZ = dA * self.activation.prime_function(self.Z)
        dW = (1/m) * np.dot(dZ, A_p.T)  # Added normalization
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)  # Added normalization
        dA_prev = np.dot(self.weights.T, dZ)
        return dA_prev, dW, db
    

class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, x):
        self.mask = np.random.rand(*x.shape) < self.rate
        return x * self.mask
    
    def backward(self, dA):
        return dA * self.mask