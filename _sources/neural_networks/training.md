---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Training Neural Networks


# Basic Feed-Forward Network with Pytorch
```{code-cell}
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# define activation functions:
activation_functions = {
    'Sign Function (Perceptron)': lambda x : np.sign(x),
    'Sigmoid Function': lambda x : 1/(1 + np.exp(-x)),
    'Hyperbolic Tangent': lambda x : np.tanh(x),
    'Rectified Linear Unit (ReLU)': lambda x : np.maximum(0,x),
    'Leaky ReLU': lambda x : np.maximum(0.1*x, x),
    'Sigmoid Linear Unit (SiLU)': lambda x : x / (1 + np.exp(-x))
}

# plot each activation function:
plt.figure(figsize=(8,6))
for i, (name,sigma) in enumerate(activation_functions.items()):
    plt.subplot(3,2,i+1)
    x = np.linspace(-4, 4, 1000)
    y = sigma(x)
    plt.grid()
    plt.xlim(-4,4)
    plt.plot(x,y)
    plt.title(name)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\sigma(x)$')

plt.tight_layout()
plt.show()
```

## Feed-Forward neural network:

```{code-cell}
import torch
import torch.nn as nn

# Define the neural network class
class FeedForwardNN(nn.Module):
    """
    torch.nn Module for a standard feed-forward
    neural network with a single hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Constructs a simple feed-forward neural network """
        super(FeedForwardNN, self).__init__()

        # Define layer 1 weights (input -> hidden layer):
        self.layer1_weights = nn.Linear(input_size, hidden_size)

        # define activation function:
        self.relu = nn.ReLU()

        # define layer 2 weights (hidden layer -> output layer):
        self.layer2_weights = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """ computes the neural network prediction """
        x2 = self.fc1(x)
        x3 = self.relu(x2)
        out = self.fc2(out)
        return out
```

```{code-cell}
# Example usage
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of the feed-forward neural network
model = FeedForwardNN(input_size, hidden_size, output_size)

# Create a random normally-distributed input array:
input_tensor = torch.randn(32, input_size)

# Forward pass
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)
```

