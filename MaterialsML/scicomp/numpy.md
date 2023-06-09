---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Numpy Package

Numpy, short for Numerical Python, is a fundamental package for scientific computing in Python. It provides efficient data structures and functions for working with multi-dimensional arrays and matrices. It also provides an interface to a large set of mathematical functions that can perform computations on arrays of data.

Another key feature of Numpy is that it provides tools to interface with low-level languages like C and FORTRAN, making it much more efficient than raw Python code. if optimized correctly, code written in Numpy can execute almost as fast as code written in these low-level languages.

## Numpy Arrays

The most important feature of the `numpy` package is that is supports _array-based_ programming, where data is organized into multi-dimensional arrays and matrices. To construct a numpy array, we call `np.array` on a Python list (or nested Python lists) as follows:

```{code-cell}
import numpy as np

np_array = np.array([1,2,3,4])

print(np_array)
print(type(np_array))
```

Numpy arrays are technically instances of the `numpy.ndarray` class, which has an instance variable `shape` that stores a tuple representing the shape of the array:

```{code-cell}
# create a 1D array:
x = np.array([1.0, 2.0, 3.0, 4.0])
print(x)
print(x.shape)

# create a 2D array (matrix):
X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(X)
print(X.shape)
```

There is no bound on the number of dimensions that a Numpy array can have. When organizing data, sometimes it is necessary to change the shape of the array. This can be done with the [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html) method:

```{code-cell}
:tags: [hide-output]
# print original shape of X:
print('X shape:', X.shape)

# reshape X to a 1D array:
X2 = X.reshape((9,))
print('X reshaped to (9,):')
print(X2)
print(X2.shape)

# reshape X to a 1x9 array:
X3 = X.reshape((1,9))
print('X reshaped to (1,9):')
print(X3)
print(X3.shape)

# reshape X to a 9x1 array:
X4 = X.reshape((9,1))
print('X reshaped to (9,1):')
print(X4)
print(X4.shape)
```

Numpy also provides some functions that can easily build arrays of of different sizes. Perhaps the most useful of these is the [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) function, which constructs a grid of evenly spaced points.

```{code-cell}
# construct a 3x3 identity (i.e. I or "eye") matrix:
print(np.eye(3))

# construct a matrix of ones:
print(np.ones((2,3)))

# construct a matrix of zeros:
print(np.zeros((3,5)))

# construct a matrix of 11 equally spaced points from 0 to 1:
x = np.linspace(0.0, 1.0, 11)
print(x)

# construct a matrix of ones with the same shape as x:
x_ones = np.ones_like(x)
x_zeros = np.zeros_like(x)
print(x_ones)
print(x_zeros)
```

## Indexing Numpy Arrays

Numpy also has some built-in syntax for accessing and modifying elements in arrays. This is similar to Python lists, but much more powerful. Below, we give some examples of how Numpy array indexing works:

```{code-cell}
X = np.array(range(1,10)).reshape((3,3))
print(X)

# access row 0:
print('\nX[0]:')
print(X[0])

# access row 0, column 2:
print('\nX[0,2]:')
print(X[0,2])

# access column 0:
print('\nX[:,0]')
print(X[:,0])
```

We can also modify the values of an array as follows:

```{code-cell}
# create an array of zeros:
Z = np.zeros((3,3))
print('Before:')
print(Z)

# modify index (1,1)
Z[1,1] = 1.0

# set column 0 to be all 3s:
Z[:,0] = 3.0
Z[2] = 4.0
```

## Operations on Numpy Arrays



## Matrix Operations


