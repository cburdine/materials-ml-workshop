# Mathematics Review

In this section, we will do a brief review of the following concepts from linear algebra and multivariate calculus:

* Vectors
* Matrices
* Matrix Operations
* Vector Operations
* Matrix-Vector Products
* Matrix Multiplication
* Eigenvalues and Eigenvectors
* Partial Derivatives
* Gradient of scalar-valued functions

## Vectors

A vector is a mathematical object with both direction and magnitude. Typically, we represent vectors as an ordered list of scalar values, where each scalar corresponds to the magnitude in a specified direction (i.e. the $x$, $y$, and $z$ directions). We typically write these scalar components as a column of numbers in square brackets. For example, the vector $\mathbf{a}$ with components in the $x$, $y$, and $z$ direction takes the form:

$$\mathbf{a} = \begin{bmatrix} a_x \\ a_y \\ a_z \end{bmatrix}$$

For higher dimensions, we typically assign a number to each dimensional component:

$$\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix}$$

## Matrices

A matrix can be thought of as an ordered list of vectors (a "vector of vectors"). Typically matrices are written as a 2D rectangular grid of scalar values, where each column of the matrix corresponds to one of these vectors. For example, a Matrix that contains 3 vectors in the $x,y,z$ basis can be written as:

$$\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}$$

A matrix is a _square_ matrix if it has the same number of rows as columns. One of the most important square matrices is the [_identity_ matrix](https://en.wikipedia.org/wiki/Identity_matrix), which is a matrix of ones along the diagonal. For example, the identity matrix in three dimensions is:

$$\mathbf{I} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

## Matrix Operations

There are several different operations that can be performed on one or more matrices. A simple operation that can be performed on one matrix is called the matrix [_transpose_](https://en.wikipedia.org/wiki/Transpose), which is denoted by a superscript $T$. The transpose operation reflects a matrix over its diagonal such that columns become rows and rows become columns. For example, using the $\mathbf{A}$ matrix from before, the transpose is:

$$\mathbf{A}^T = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}^T = \begin{bmatrix}
a_{11} & a_{21} & a_{31} \\
a_{12} & a_{22} & a_{32} \\
a_{13} & a_{23} & a_{33}
\end{bmatrix}$$

We can multiply a scalar times a matrix, which is commonly denoted by juxtaposing the scalar in front of the matrix. For example, multiplying a scalar $c$ times $\mathbf{A}$, we get:

$$c \mathbf{A} = \begin{bmatrix}
c \cdot a_{11} & c \cdot a_{12} & c \cdot a_{13} \\
c \cdot a_{21} & c \cdot a_{22} & c \cdot a_{23} \\
c \cdot a_{31} & c \cdot a_{32} & c \cdot a_{33}
\end{bmatrix}$$

If two matrices have the same shape, we can add or subtract them by performing the addition elementwise:

$$\mathbf{A} \pm \mathbf{B} = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix} \pm \begin{bmatrix}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33}
\end{bmatrix} = \begin{bmatrix}
a_{11} \pm b_{11} & a_{12} \pm b_{12} & a_{13} \pm b_{13} \\
a_{21} \pm b_{21} & a_{22} \pm b_{22} & a_{23} \pm b_{23} \\
a_{31} \pm b_{31} & a_{32} \pm b_{32} & a_{33} \pm b_{33}
\end{bmatrix}$$

Since vectors are essentially the same as a matrix with one column, addition, subtraction, and multiplication by vectors behaves essentially the same with vectors as it does with matrices:

$$c\mathbf{a} = \begin{bmatrix} 
c \cdot a_1 \\
c \cdot a_2 \\
c \cdot a_3
\end{bmatrix},\qquad \mathbf{a} \pm \mathbf{b} = \begin{bmatrix}
a_1 \pm b_1 \\ a_2 \pm b_2 \\ a_3 \pm b_3 
\end{bmatrix}$$

For any two vectors with the same number of dimensions, there are several different kinds of "vector multiplication" operations defined. In this review we will discuss two of these "multiplication" operations: the vector _inner product_ (also called the _dot_ product) and the vector _outer product_.

### Vector Inner Product

The [inner product](https://en.wikipedia.org/wiki/Dot_product) of two vectors (also called the dot product) is a scalar quantity that described the degree to which any two vectors are "aligned" with eachother. For two vectors $\mathbf{a}$ and $\mathbf{b}$ the inner product often denoted either as $\mathbf{a} \cdot \mathbf{b}$ or $\mathbf{a}^T\mathbf{b}$ (the reason for the transpose superscript $T$ will become clear later). The inner product is defined as the sum of the product of the corresponding scalar entries of both vectors, namely:

$$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T\mathbf{b} = \begin{bmatrix} a_1 & a_2 & \dots & a_N \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ a_N \end{bmatrix} = a_1b_1 + a_2b_2 + \dots + a_Nb_N$$

The other type of vector product, the [_outer_ product](https://en.wikipedia.org/wiki/Outer_product) evaluates to a matrix of values representing the product of every pair of components. The outer product is typically denoted as $\mathbf{a}\mathbf{b}^T$:

$$\mathbf{a}\mathbf{b}^T = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_N \end{bmatrix} \begin{bmatrix} b_1 & b_2 & \dots & a_N \end{bmatrix} = \begin{bmatrix}
a_1b_1 & a_1b_2 & \dots  & a_1 b_N \\
a_2b_1 & a_2b_2 & \dots  & a_2 b_N \\
\vdots & \vdots & \ddots & \vdots  \\
a_Nb_1 & a_Nb_2 & \dots  & a_Nb_N
\end{bmatrix}$$

### Matrix-Vector Products

We can generalize the inner product of two vectors to the matrix-vector product. Specifically, the product of a matrix $\mathbf{A}$ and a vector $\mathbf{x}$ is a vector consisting of the dot products of the corresponding rows of $\mathbf{A}$ with the vector $\mathbf{x}$. The product of a matrix with a vector is often denoted by juxtaposition:

$$\mathbf{A}\mathbf{x} = \begin{bmatrix}
- ~ \mathbf{a}_1^T ~ - \\
- ~ \mathbf{a}_2^T ~ - \\
    \vdots             \\
- ~ \mathbf{a}_M^T ~ - \\
\end{bmatrix} \begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_N
\end{bmatrix} = \begin{bmatrix}
\mathbf{a}_1^T\mathbf{x} \\
\mathbf{a}_2^T\mathbf{x} \\
\vdots \\
\mathbf{a}_N^T\mathbf{x}
\end{bmatrix} = \begin{bmatrix}
a_{11}x_1 + a_{12}x_2 + \dots + a_{1N}x_N \\
a_{21}x_1 + a_{22}x_2 + \dots + a_{2N}x_N \\
\vdots \\
a_{M1}x_M + a_{M2}x_2 + \dots + a_{MN}x_N
\end{bmatrix}$$

From the definition of the matrix-vector product, we see that the number of columns of $\mathbf{A}$ must equal the number of dimensions of $\mathbf{x}$. Thus, if $\mathbf{A}$ is an $M \times N$ matrix (i.e. with $M$ rows and $N$ columns), the matrix-vector product $\mathbf{Ax}$ only exists with vectors $\mathbf{x}$ of dimension $N$.

### Systems of Linear Equations

One important application of the matrix-vector product is in solving systems of linear equations. Consider a system of $M$ linear equations involving $N$ variables $x_1, x_2, ..., x_N$:

$$\begin{aligned}
a_{11}x_1 + a_{12} x_2 + \dots + a_{1N}x_{N} &= b_{1} \\
a_{21}x_1 + a_{22} x_2 + \dots + a_{2N}x_{N} &= b_{2} \\
\vdots \qquad\qquad\qquad &\qquad \vdots \\
a_{M1}x_1 + a_{M2} x_2 + \dots + a_{MN}x_{N} &= b_{M} \\
\end{aligned}$$

Above, the coefficients $a_{ij}$ and $b_j$ are known constants, but the values of the variables $x_i$ must be solved for. We can re-write the system of equations above as an equation involving a matrix-vector product, namely:

$$\begin{bmatrix}
a_{11} & a_{12} & \dots  & a_{1N} \\
a_{21} & a_{22} & \dots  & a_{2N} \\
\vdots & \vdots & \ddots & \vdots  \\
a_{M1} & a_{M2} & \dots  & a_{MN}
\end{bmatrix}\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_N
\end{bmatrix} = \begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ x_M
\end{bmatrix}\mathbf{Ax} = \mathbf{b}$$

Depending on the entries of $\mathbf{A}$ and $\mathbf{b}$, the equation $\mathbf{Ax} = \mathbf{b}$ may have no solution, a unique solution, or infinitely many solutions. In the special case where the number of equations equals the number of variables (i.e. $N = M$), then it can be shown that a unique solution $\mathbf{x}$ exists for this system if and only if a quantity called the [_determinant_ of $\mathbf{A}$](https://en.wikipedia.org/wiki/Determinant) (denoted $\det(\mathbf{A})$) is non-zero. If $\det(\mathbf{A}) \neq 0$, then the unique solution to $\mathbf{Ax} = \mathbf{b}$ is given by the matrix-vector product ($\mathbf{A}^{-1})\mathbf{b}$, where $\mathbf{A}^{-1}$ is a matrix that is called the [_inverse_](https://en.wikipedia.org/wiki/Invertible_matrix) of $\mathbf{A}$.

The inverse matrix $\mathbf{A}^{-1}$ does not have a closed form solution (at least, not one that is easy to compute by hand), but there are ways of computing it numerically (for example, the `numpy` Python package has the [`np.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) subroutine).

## Matrix Products

We can further generalize the matrix-vector product to the product of two matrices. We denote the product of two (or more) matrices by juxtaposition (i.e. $\mathbf{AB}$). The product of two matrices is simply a matrix whose columns are the the result of the product of the left matrix times the corresponding column of the right matrix. 

Namely:

$$\mathbf{A}\mathbf{B} = \mathbf{A}\begin{bmatrix}
| & | & & | \\
\mathbf{b_1} & \mathbf{b_2} & \dots & \mathbf{b_N} \\
| & | & & | \\
\end{bmatrix} = \begin{bmatrix}
| & | & & | \\
(\mathbf{Ab_1}) & (\mathbf{Ab_2}) & \dots & (\mathbf{Ab_N}) \\
| & | & & | \\
\end{bmatrix}$$

This product can only be computed if the number of columns of $\mathbf{A}$ equals the number of rows of $\mathbf{B}$. Specifically, if $\mathbf{A}$ has $L$ rows and $M$ columns (i.e. $\mathbf{A}$ is an $L \times M$ matrix) and $\mathbf{B}$ has $M$ rows and $N$ columne (i.e. $\mathbf{B}$ is an $M \times N$ matrix$), the resulting matrix $\mathbf{AB}$ is $L \times N$.

If we expand $\mathbf{A}$ in terms of its rows $\mathbf{a_1}^T, \mathbf{a_2}^T, ..., \mathbf{a_L}^T$ we see that the entries of the matrix resulting from the product $\mathbf{AB}$ consist of inner products of the rows of $\mathbf{A}$ and the columns of $\mathbf{B}$:

$$\mathbf{AB} = \begin{bmatrix}
- ~ \mathbf{a}_1^T ~ - \\
- ~ \mathbf{a}_2^T ~ - \\
    \vdots             \\
- ~ \mathbf{a}_L^T ~ - \\
\end{bmatrix}\begin{bmatrix}
| & | & & | \\
\mathbf{b_1} & \mathbf{b_2} & \dots & \mathbf{b_N} \\
| & | & & | \\
\end{bmatrix} = \begin{bmatrix}
\mathbf{a_1}^T\mathbf{b_1} & \mathbf{a_1}^T\mathbf{b_2} & \dots & \mathbf{a_1}^T\mathbf{b_N} \\
\mathbf{a_2}^T\mathbf{b_1} & \mathbf{a_2}^T\mathbf{b_2} & \dots & \mathbf{a_2}^T\mathbf{b_N} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a_L}^T\mathbf{b_1} & \mathbf{a_L}^T\mathbf{b_2} & \dots & \mathbf{a_L}^T\mathbf{b_N}
\end{bmatrix}$$

From the definition of the inner product, it can be shown that the entry in row $i$ and column $k$ of $\mathbf{AB}$ is a sum of $N$ products of corresponding entries of the rows of $\mathbf{A}$ and the columns $\mathbf{B}$. Specifically:

$$(\mathbf{AB})_{ik} = \mathbf{a}_i^T\mathbf{b}_k = \sum_{j=1}^N a_{ij}b_{jk}$$

## Eigenvalues and Eigenvectors

If a matrix $\mathbf{A}$ is square (i.e. it an $N \times N$ matrix, having $N$ rows and $N$ columns), then it can be shown that $\mathbf{A}$ has a set of $N$ scalar values associated with it called [_eigenvalues_](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors). These eigenvalues (often denoted $\lambda_1, \lambda_2, ... \lambda_N$) are the values that satisfy the equation:

$$\mathbf{Au} = \lambda\mathbf{u}$$

for a certain set of vectors $\mathbf{u}$. These vectors $\mathbf{u}$ that satisfy this equation for each eigenvalue $\lambda_i$ are the [_eigenvectors_](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) associated with the eigenvalue $\lambda_i$. Although we will not give a rigorous summary of eigenvalues and eigenvectors here, we remark that these quantities play a significant role in many fields such as mathematics, statistics, and quantum mechanics.

## Partial Derivatives

Let $f(x_1, x_2, ..., x_N)$ be a real scalar-valued function of $N$ real variables. In this instance, we can equivalently think of $f$ as a scalar function of an $N$-dimensional vector $\mathbf{x}$ in the vector space $\mathbb{R}^N$, that is, we write $f(\mathbf{x}) : \mathbb{R}^N \rightarrow \mathbb{R}$.

For such a function $f$, the _partial derivative_ of $f$ with respect to a component $x_i$ (written $\partial f/\partial x_i$) is defined as the derivative of $f$ in the direction of the component $x_i$. Algebraically, we compute the partial derivative just like we compute single-variable derivatives, except for the fact that all other variables ($x_j$ where $j \neq i$) are treated as constants such that $dx_j  / dx_i = 0$: 

$$\dfrac{\partial f}{\partial x_i} \equiv \dfrac{d}{dx_i} \left[ f\right]_{x_{j \neq i} \text{ all constant }}$$

Below, we give some examples of partial derivatives for functions of three dimensions where $(x_1,x_2,x_3) = (x,y,z)$:

* $f(x,y,z) = 3x + 4y + 2$: 
    - $\frac{\partial f}{\partial x} = 3$ 
    - $\frac{\partial f}{\partial y} = 4$ 
    - $\frac{\partial f}{\partial z} = 0$

* $f(x,y,z) = xy/z$:
    - $\frac{\partial f}{\partial x} = y/z$
    - $\frac{\partial f}{\partial y} = x/z$
    - $\frac{\partial f}{\partial z} = -xy/z^2$

* $f(x,y,z) = \sin(xyz)$:
    - $\frac{\partial f}{\partial x} = yz\cos(xyz)$
    - $\frac{\partial f}{\partial y} = xz\cos(xyz)$
    - $\frac{\partial f}{\partial z} = xy\cos(xyz)$

## Gradient of scalar-valued functions

For a scalar valued function of a vector $\mathbf{x}$, that is, $f: \mathbb{R}^N \rightarrow \mathbb{R}$, we can compute a vector-valued function called the _gradient of $f$_. The gradient of $f$, denoted by $\nabla f$, is the vector of partial derivatives of $f$ with respect to each component of $\mathbf{x}$:

$$ \nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \dots & \frac{\partial f}{\partial x_N} \end{bmatrix}^T $$

If we evaluate the gradient at a point $\mathbf{x}$ it results in a vector that "points" in the direction of the greatest increase of $f$. The magnitude of $\nabla f$ corresponds to the slope of $f$ in this direction of greatest increase.

## Exercises

:::{dropdown} Exercise 1: The Matrix Inverse
(TODO)
:::

:::{dropdown} Exercise 2: Eigenvalues
(TODO)
:::
