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

Another key feature of Numpy is that it interfaces with low-level languages like C and FORTRAN, making it much more efficient than raw Python code. If optimized correctly, code written with Numpy can execute almost as fast as code written in these low-level languages.

## Numpy Arrays

The most important feature of the `numpy` package is that is supports _array-based_ programming, where data is organized into multi-dimensional arrays and matrices. To construct a Numpy array, we call `np.array` on a Python list (or nested Python lists) as follows:

```{code-cell}
import numpy as np

np_array = np.array([1,2,3,4])

print(np_array)
print(type(np_array))
```

Numpy arrays are technically instances of the `numpy.ndarray` class, which has an instance variable `shape` that stores a tuple representing the shape of the array. The length of `shape` corresponds to the number of dimensions of the array, while the product of the elements in `shape` corresponds to the total number of elements in the array:

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

When organizing data, sometimes it is necessary to change the shape of the array. This can be done with the [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html) method:

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

# construct an array of ones:
print(np.ones((2,3)))

# construct an array of zeros:
print(np.zeros((3,5)))

# construct an array of 11 equally spaced points from 0 to 1:
x = np.linspace(0.0, 1.0, 11)
print(x)

# construct an array of ones/zeros with the same shape as x:
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
print('Accessing X[0]:')
print(X[0])

# access row 0, column 2:
print('Accessing X[0,2]:')
print(X[0,2])

# access column 0:
print('Accessing X[:,0]:')
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

# set column 0 to be all 2s:
Z[:,0] = 2.0

# set row 2 to be the following values:
Z[2] = np.array([3.0, 4.0, 5.0])

# print resulting array:
print('After:')
print(Z)
```

## Operations on Numpy Arrays

When we apply mathematical operations (i.e. `+`,`-`,`*`, etc.) to Numpy arrays with a compatible shape, the result is a Numpy array where the operation is performed elementwise. For example:

```{code-cell}
x1 = np.array([1,2,3])
x2 = np.array([4,5,6])

# all math operations are elementwise:
print(x1 + x2)
print(x1 - x2)
print(x1 * x2)

# scalar operations are also elementwise:
print('\nScalar operations:')
print(x1 - 1)
print(x1 * 2)
print(x1**2)
print(np.sqrt(x1))
print(-x1)
```

:::{note}
In order for an operation to be applied to two Numpy arrays, the `shape` of one array must be _broadcastable_ to the other. Arrays with the same `shape` are always broadcastable. To learn more about what _broadcastable_ means, see the [Numpy tutorial on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
:::

## Reducing Operations:

Numpy supports reduction operations such as `min`, `max`, `mean` and `sum`, which can be applied to all elements in an array or only along a specified dimension. For example:

```{code-cell}
# create example matrix:
X = np.array(range(9)).reshape(3,3)
print('X array:')
print(X)

# compute the mean of all elements:
print(np.mean(X))

# compute the mean along each column:
print(np.mean(X, axis=0))

# compute the sum along each row:
print(np.sum(X, axis=1))

# compute the min/max values:
print(np.min(X), np.max(X))

```

## Matrix Operations

For 2D Numpy array that represents a [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)), Numpy also supports many different operations in its `linalg` package. For example:

```{code-cell}
:tags: [hide-output]
# generate a matrix:
A = np.array(range(1,10)).reshape(3,3)
print(A)

# generate a diagonal matrix:
X = np.diag([1,10,100])
print(X)

# matrix transpose:
print(A.T)

# matrix inverse:
print(np.linalg.inv(X))
```

:::{important}
If you are in need of a review of matrices, vectors, and other linear algebra content, see the {doc}`../ml_intro/math_review` Section.
:::

We can also compute matrix-matrix products, matrix-vector products, and vector-vector products with the reserved `@` operator:

```{code-cell}
:tags: [hide-output]
# compute a matrix-matrix product:
print('A @ X:')
print(A @ X)

# compute a matrix-vector product:
b = np.array([1,10,100])
print('A @ b:')
print(A @ b)

# compute a vector dot product (inner product):
print('b dot b:')
print(np.dot(b,b))

# compute vector outer product:
print('b outer b:')
print(np.outer(b,b))

# compute vector norm:
print('|b|:')
print(np.linalg.norm(b))
```

Some other useful matrix operators are supported, such as solvers for eigenvalues, eigenvectors, and determinants. Kronecker products are also supported:

```{code-cell}
:tags: [hide-output]
# eigenvalue decomposition of a square matrix:
# (if A is symmetric, use linalg.eigh instead)
eigvals, eigvects = np.linalg.eig(A)
print(eigvects)
print(eigvals)

# matrix determinant:
print(np.linalg.det(A))

# Kronecker (tensor) product:
np.kron(A,np.diag([1,10]))
```

## Exercises

:::{dropdown} Exercise 1: Solving a Linear System
Write some Python that solves for the vector $\mathbf{x}$ in the matrix equation $\mathbf{A}\mathbf{x} = \mathbf{b}$, where $\mathbf{A}$ is a square matrix stored in a Numpy array `A` and $\mathbf{b}$ is a vector stored in a Numpy array `b`. Verify that $\mathbf{x}$ is the correct solution by computing $\mathbf{A}\mathbf{x}$ and comparing the result with $\mathbf{b}$.

---
_Hint_: Recall from linear algebra that the solution to a linear system is $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$, so you may need to use the [`np.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) function.

:::

:::{dropdown} Exercise 2: Eigendecomposition:
Here's an exercise for the physicsts:

Any square matrix $\mathbf{A}$ that is [non-defective](https://en.wikipedia.org/wiki/Defective_matrix) can be written in the form:

$$\mathbf{A} = \mathbf{P} \Lambda \mathbf{P}^{-1}$$

where $\Lambda$ is a diagonal matrix containing the eigenvalues of $\mathbf{A}$, and $\mathbf{P}$ is a matrix whose columns are the corresponding eigenvalues of $\mathbf{A}$. When $\mathbf{A}$ is factorized in this form it is called an [_eigendecomposition_](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Eigendecomposition_of_a_matrix), and it is an important result that is often used in physics and quantum mechanics. Do the following:


First, solve for $\mathbf{P}$ and $\Lambda$ ising [`np.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) for the following matrix:

$$\mathbf{A} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

Using the `@` operator, verify numerically that the identity above holds. Does the largest entry of $\Lambda$ [look familiar](https://en.wikipedia.org/wiki/Golden_ratio)?

Next, do the same thing, but with the matrix:

$$\mathbf{H} = \begin{bmatrix} 0 & 1 & 0\\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

Since $\mathbf{H}$ is symmetric (more generally, Hermitian), use [`np.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) instead of `np.linalg.eig` to get better numerical precision.
:::

### Solutions

#### Exercise 1: Solving a Linear System:

```{code-cell}
:tags: [hide-cell]
# dimension of the system:
N = 4

# generate a random  NxN matrix A:
A = np.random.normal(0,1,size=(N,N))

# generate a random 10x10 target vector b:
b = np.random.normal(0,1,size=N)

# print A and b:
print('A:')
print(A)
print('b:', b)

# solve for x:
x = np.linalg.inv(A) @ b

# print x:
print('x:', x)

# print Ax (for comparison with b):
print('Ax:', A @ x)
```

#### Exercise 2: Eigendecomposition:

```{code-cell}
:tags: [hide-cell]

#------------------------------------
# Part one:
#------------------------------------

# given matrix A:
A = np.array([ [1,1], [1,0] ])

# diagonalize A:
lambda_diag, P = np.linalg.eig(A)
Lambda = np.diag(lambda_diag)

# print P:
print('P:')
print(P)

# print Λ:
print('Λ:')
print(Lambda)

# convert the eigenvalues into a diagonal matrix:
Lambda = np.diag(lambda_diag)

print('P Λ P^(-1):')
print(P @ Lambda @ np.linalg.inv(P))


#------------------------------------
# Part Two:
#------------------------------------
print('----------------------------')

# given matrix X:
X = np.array([
    [ 0, 1, 0 ],
    [ 1, 0, 1 ],
    [ 0, 1, 0 ]
])

# diagonalize X using eigh (for Hermitian matrices):
lambda_diag, P = np.linalg.eigh(X)
Lambda = np.diag(lambda_diag)

# print P:
print('P:')
print(P)

# print Λ:
print('Λ:')
print(Lambda)

# convert the eigenvalues into a diagonal matrix:
Lambda = np.diag(lambda_diag)

print('P Λ P^(-1):')
print(P @ Lambda @ np.linalg.inv(P))

```
