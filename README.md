# Neural Network Implementation from Scratch

A pure NumPy implementation of a deep neural network, built for educational purposes and deep learning understanding. This project implements a multi-layer neural network with various features commonly found in modern deep learning frameworks.

## Features

- **Pure NumPy Implementation**: No deep learning frameworks used, everything built from scratch
- **Modular Architecture**: Separate classes for layers, activations, optimizers, and loss functions
- **Supported Components**:
  - Dense (Fully Connected) Layers
  - Dropout Layers for regularization
  - ReLU and Sigmoid activation functions
  - SGD optimizer with gradient clipping
  - Binary Cross-entropy loss
  - Various metrics calculations (Accuracy, Precision, Recall, F1-score)

## Project Structure

```
├── neuralnet.py      # Main neural network implementation
├── layers.py         # Layer implementations (Dense, Dropout)
├── activation.py     # Activation functions
├── optimizer.py      # Optimization algorithms
├── loss.py          # Loss functions
├── metrics.py       # Evaluation metrics
└── implement.ipynb   # Example usage notebook
```

## Usage Example

```python
from neuralnet import NeuralNet
from activation import ReLU, Sigmoid
from optimizer import SGD
from loss import Loss

# Create network
nn = NeuralNet()
nn.dense(784, 128, ReLU())
nn.dense(128, 64, ReLU())
nn.dense(64, 10, Sigmoid())

# Compile
nn.compile(SGD(learning_rate=0.01), Loss.binary_crossentropy)

# Train
history = nn.fit(x_train, y_train, 
                epochs=10, 
                batch_size=32, 
                val_split=0.2, 
                verbose=2)

# Predict
predictions = nn.predict(x_test)
```

## Features in Detail

### Layers
- **Dense Layer**: Fully connected layer with configurable input/output dimensions
- **Dropout Layer**: Regularization layer with configurable dropout rate

### Activation Functions
- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Sigmoid activation for binary classification

### Optimizers
- **SGD**: Stochastic Gradient Descent with gradient clipping

### Loss Functions
- **Binary Cross-entropy**: For classification tasks

### Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Requirements

- NumPy
- scikit-learn (for metrics and data splitting)
- tqdm (for progress bars)
- matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -r requirements.txt
```

## Example Results

The network has been tested on the MNIST dataset, achieving reasonable performance for a from-scratch implementation:
- Training accuracy: ~XX%
- Validation accuracy: ~XX%

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built as an educational project to understand deep learning fundamentals
- Inspired by modern deep learning frameworks like TensorFlow and PyTorch

