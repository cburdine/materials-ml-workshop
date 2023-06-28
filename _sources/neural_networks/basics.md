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

# Basic Neural Networks

In this section we will describe in greater detail how a basic neural network model works and why a neural network has greater flexibility than some of the models we have studied previously. This flexibility, however, often comes at the cost of low interpretability, as generating a simple explanation of why a large neural network makes a particular prediction is often quite difficult.

In this section we will learn some basic principles about neural networks and then implement a densely-connected feed-forward neural network using the popular machine learning framework Pytorch.

## A Single Neuron

In previous sections, we encountered the linear classifier model (or Perceptron), which had the following form:

$$f(\mathbf{x}) = \text{sign}\left( w_0 + \sum_{i=1}^D w_ix_i \right)$$

where $\text{sign}(x) = \begin{cases} +1 &  x > 0\\ -1 & x <= 0 \end{cases}$.

Originally, the perceptron model was inspired by the biological function of a [multipolar neuron](https://en.wikipedia.org/wiki/Multipolar_neuron), which produces an electrical response (the "output" of the neuron) if a weighted sum of electrical stimuli from neighboring neurons (the "inputs" of the neuron) exceed a given threshold. In this model of a neuron, the function that dictates the neuron's response with respect to the sum of inputs is referred to as a _neuron activation function_, which we will denote here as $\sigma(x)$. This activation function is applied before the neuron outputs a response, as shown in the diagram below:

![neuron](neuron.svg)

More generally, the output of a single neuron with weight vector $\mathbf{w} = \begin{bmatrix} w_0 & w_1 & \dots &  w_D \end{bmatrix}^T$ and activation function $\sigma(x)$ can be written as follows:

$$f(\mathbf{x}) = \sigma(\mathbf{w}^T\underline{\mathbf{x}}) = \sigma\left( w_0 + \sum_{i=1}^D w_ix_i \right)$$

(Recall that $\underline{\mathbf{x}}$ is the vector $\mathbf{x}$ prependend with $1$: $\underline{\mathbf{x}} = \begin{bmatrix} 1 & x_1 & x_2 & \dots & x_D \end{bmatrix}$)

## Activation Functions:

In order for a network of neurons to "learn" from non-linear data, it is critical that the neuron activation function $\sigma(x)$ is non-linear. For example, in the perceptron model, the activation function is $\sigma(x) = \text{sign}(x)$, which fits this criterion. However, this function is not continuous, and its derivative is $0$ almost everywhere. In order to fit a neural network to data though a method such as gradient descent, it is desirable that the activation function $\sigma(x)$ be both continuous and differentiable. Below, we give some alternative activation functions that are commonly used in neural networks:

* _Sigmoid function_: 

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

* _Hyperbolic Tangent_: 

$$\sigma(x) = \tanh(x)$$

* _Rectified Linear Unit (ReLU)_: 

$$\sigma(x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}$$

* _Leaky ReLU_: 

$$\sigma(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0\end{cases}$$

($\alpha$ is chosen such that $0 < \alpha \ll 1$. Typically, $\alpha = 10^{-3}$)

* _Sigmoid Linear Unit (SiLU)_: 

$$\frac{x}{1 + e^{-x}}$$

The activation function used is ofte chosen depending on the kind of outputs desired for each neuron and the kind of model being used. In most cases, the ReLU activation function is a good choice.

Below, we write some Python code that visualizes each of these activation functions:

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
## Networks of Neurons

By networks of individual neurons into layers and stacking these layers, we can produce some very powerfule non-linear models. Layered neural network models can be applied to almost any supervised learning task, even tasks where the there are multiple labels that need to be predicted (i.e. where $\mathbf{y}$ is a vector, not just a scalar).

The simplest kind of neural network layer we can construct is a _fully-connected_ layer, in which a collection of neurons produce a vector of outputs $\mathbf{a}$ (where each element $a_i$ corresponds to a single neuron output) based on different linear combinations of the input features. Specifically, a layer of $m$ neurons can be used to compute the function:

$$\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} = f(\mathbf{x}) = \begin{bmatrix} 
\sigma\left(w_{1,0} + \sum_{i=1}^D w_{1,i}x_i\right) \\ 
\sigma\left(w_{2,0} + \sum_{i=1}^D w_{2,i}x_i\right) \\
\vdots \\
\sigma\left(w_{m,0} + \sum_{i=1}^D w_{m,i}x_i\right)
\end{bmatrix}$$

Above $w_{i,j}$ (sometimes written $w_{ij}$) denotes weight of feature $j$ in neuron $i$. In total, this layer of neurons has $m \times (D+1)$ weights, which we can organize into a rectangular weight matrix $\mathbf{W}$ as follows:

$$\mathbf{W} = \begin{bmatrix}
w_{10} & w_{11} & \dots & w_{1D} \\
w_{20} & w_{12} & \dots & w_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m0} & w_{m1} & \dots  & w_{mD}
\end{bmatrix}$$

In terms of this weight matrix, we can write the layer's function as a matrix-vector product, namely:

$$\mathbf{a} = f(\mathbf{x}) = \sigma(\mathbf{W}\mathbf{x})$$

(Note that when we write $\sigma(\mathbf{A})$ where $\mathbf{A}$ is a matrix or vector, it denotes that the activation function $\sigma$ is applied element-wise; that is, to each entry of $\mathbf{A}$ individually.)


A _standard feed-forward neural network_ is a simple kind of neural network that stacks two layers of neurons; the first layer computes a vector of features $\mathbf{a}$ from the data $\mathbf{x}$. This layer is sometimes called a _hidden layer_. Next, a second layer computes the output vector $\hat{\mathbf{y}}$. (In the case where only a single output is desired, only one neuron is used in the output layer to output a scalar value $\hat{y}$.) Here, we will let $\mathbf{W}$ denote the weight matrix of the hidden layer, and $\mathbf{V}$ denote the weight matrix of the second layer. We can visualize this network and the connectedness between neuron layers as follows:

![FeedForward Neural Network](simple_nn.svg)

By composing the equations for the hidden layer inside the equation for the output layer, we obtain the final model equation for a standard feed-forward neural network:

$$f(\mathbf{x}) = \sigma\left( \mathbf{V}\underline{\sigma\left(\mathbf{W}\underline{\mathbf{x}}\right)} \right)$$
