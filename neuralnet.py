from layers import Dense, Dropout
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from metrics import Metrics

class NeuralNet:
    def __init__(self):
        self.layers = []

    def dense(self, input_size, output_size, activation):
        self.layers.append(Dense(input_size, output_size, activation))
    
    def dropout(self, rate):
        self.layers.append(Dropout(rate))

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, x):
        x = x.reshape(x.shape[0], -1).T  # Flatten and transpose to (n_features, n_batch)
        for layer in self.layers:
            x = layer.forward(x)
        return x.T

    def backward(self, x, y):
        # Calculate initial error (y_true - y_pred)
        y_pred = self.forward(x)
        error = (y_pred - y.reshape(y.shape[0], -1)).T  # Changed from just using y as error
        
        gradients = []
        dA = error  # Start with the error
        x = x.reshape(x.shape[0], -1).T 
        
        # Collect gradients
        for i in reversed(range(len(self.layers))):
            A_prev = self.layers[i-1].A if i > 0 else x
            dA, dW, db = self.layers[i].backward(dA, A_prev)
            gradients.append((dW, db))
        
        # Update weights using optimizer
        for i, (layer, (dW, db)) in enumerate(zip(reversed(self.layers), gradients)):
            self.optimizer.update(layer, dW, db)
        
        return dA

    def fit(self, X, y, epochs = 100, batch_size = 32, val_split = 0.2, verbose = 1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split)
        for epoch in tqdm(range(epochs)):
            # Generate random indices for the entire epoch
            indices = np.random.permutation(len(X_train))
            
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if verbose == 2:
                print(f"Epoch {epoch + 1}/{epochs}: Train Loss {self.loss(y_train, self.forward(X_train))}")

        if verbose:
            print(f"Train Loss: {self.loss(y_train, self.forward(X_train))}")
            print(f"Validation Loss: {self.loss(y_val, self.forward(X_val))}")
            print(f"Validation Accuracy: {Metrics.accuracy(y_val, self.forward(X_val))}")
            print(f"Training Metrics: {Metrics.accuracy(y_train, self.forward(X_train))}")

        return Metrics.all_metrics(y_val, self.forward(X_val))
        

    def predict(self, X):
        return self.forward(X)