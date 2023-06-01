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

# Kernel Machines

In this section, we will introduce a powerful and popular class of regression and classification models known as _kernel machines_. Kernel machines are an extension of the high-dimensional embedding models of the form:

$$\hat{y} = f(\mathbf{x}) = w_0 + \sum_{i=1}^{D_{emb}} w_1\phi_i(\mathbf{x})$$ 

As we will see, kernel machines can even be used to compute functions where $D_{emb} = \infty$ (i.e. the data is embedded in an infinite-dimensional space). This is achieved through computing only the inner products between data points, given by a _kernel function_ $K(\mathbf{x}, \mathbf{x}')$.

## Maximum Absolute Error Regression:

In order to understand kernel functions, it is helpful to first examine the problem of maximum absolute error regression. The goal of this regression model is to find the simplest high-dimensional embedding model (i.e. the one with the minimal sum of weights squared) subject to the constraint that the maximum absolute error of all predictions in the training set lie within an error tolerance value $\varepsilon$. Formally, we write this optimization problem as:

$$\text{minimize: }\ \sum_i^{D_{emb}} w_i^2\ \text{ subject to: }\  \max_n |\hat{y}_n - y_n| < \varepsilon $$

This is an instance of a _Lagrange multiplier problem_, which we can re-write in standard form.

:::{admonition} Note: Lagrange Multiplier Problems
:class: note, dropdown

A Lagrange multiplier problem is an optimization problem concerning the optimization of an objective function $f$ subject to a set of constraint functions $g$. Such a problem is in _standard form_, if it takes the form:

$$\text{minimize: }\ f(\mathbf{w})\ \text{ subject to: }\ \begin{cases} g_1(\mathbf{w}) \le 0 \\ g_2(\mathbf{w}) \le 0 \\ \dots  \\ g_k(\mathbf{w}) \le 0 \end{cases}$$

The _Lagrangian_ of this problem is the function:

$$\mathcal{L}(\mathbf{w}) = f(\mathbf{w}) + \sum_{i=1}^k \lambda_i g(\mathbf{w})$$

where the $\lambda_i$ are variables called _Lagrange multipliers_. It can be shown that all local minima and maxima of $f(\mathbf{w})$ must satisfy the equation:

$$\nabla_w \mathcal{L}(\mathbf{w}) = 0$$

This equation, along with the constraints $g(\mathbf{w})$ can be used to determine the multipliers $\lambda_i$ and the associated points $\mathbf{w}$ where $f$ attains local maxima and minima.
:::

Writing the maximum absolute error regression problem in the standard form of a lagrange multiplier problemn, we obtain:

$$\begin{aligned}
\text{maximize: } &  -\frac{1}{2}(\mathbf{w}^T\mathbf{w})\\
 \text{ subject to: } & \begin{cases} 
\left(w_0 + \sum_{i=1}^{D_{emb}} w_i\phi_i(\mathbf{x}_n) - y_n\right) - \varepsilon \le 0\\
-\left(w_0 + \sum_{i=1}^{D_{emb}} w_i\phi_i(\mathbf{x}_n) - y_n\right) - \varepsilon \le 0 \\
\end{cases}
\ \text{ for $n = 1, 2, ..., N$}
\end{aligned}$$

This problem has a total of $k=2N$ contraint functions, which means that we must introduce a total of $2N$ Lagrange multipliers. Letting $\alpha_1, \alpha_2, ..., \alpha_n$ and $\alpha_1^*, \alpha_2^*, ..., \alpha_n^*$ be the Lagrange multipliers of the top and bottom set of constraints, the Lagrangian function with respect to the weights $\mathbf{w}$ becomes:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{2}(\mathbf{w}^T\mathbf{w}) + \sum_{n=1}^N \left[ -(\alpha_n - \alpha_n^*)\left(w_0 + \sum_{i=1}^{D_{emb}} w_i \phi_i(\mathbf{x}_n) - y_n\right) + (\alpha_n + \alpha_n^*)\varepsilon \right]$$

:::{important}
A couple of notational warnings for physicists: 
1. The notation $\alpha_n^*$ does not denote the complex conjugate here; the Lagrange multipliers $\alpha_n$ and $\alpha_n^*$ are independent real scalar values.
2. The "Lagrangian" function used here is unrelated to the Lagrangian operator $\mathcal{L} = \mathcal{T} - \mathcal{V}$, though both of these formalisms are attributed to [Lagrange](https://en.wikipedia.org/wiki/Joseph-Louis_Lagrange). (Same guy, different crime).
::: 

Setting the gradient of the Lagrangian equal to $\mathbf{0}$, we obtain an expression for the weights $\mathbf{w}$:

$$\nabla_w \mathcal{L}(\mathbf{w}) = \mathbf{0}\qquad \Rightarrow \qquad  w_i = \sum_{n=1}^N (\alpha_n - \alpha_n^*)\phi_i(\mathbf{x}_n)$$

To ensure a solution exists to our optimization problem, we impose the following additional constraints on the Lagrange multipliers:

$$ \sum_{n = 1}^N (\alpha_n - \alpha_n^*) = 0,\qquad   0 \le \alpha_n, \alpha_n^* \le C$$

After imposing these constraints, substituting the solution to $\nabla_w \mathcal{L}(\mathbf{w}) = 0$ for $\mathbf{w}$, and reformulating it as a minimization problem, we obtain the following [_dual formulation_](https://en.wikipedia.org/wiki/Duality_(optimization)#Dual_problem) of the maximum absolute error regression problem:

$$\begin{aligned}
\text{minimize: } &  \frac{1}{2}\mathbf{a}^T\mathbf{G}\mathbf{a} - \mathbf{a}^T\mathbf{y} + \varepsilon\left(\sum_{n=1}^N (\alpha_n + \alpha_n^*)\right)\\
 \text{ subject to: } & \begin{cases} 
\sum_{n=1}^N (\alpha_n - \alpha_n^*) = 0 \\
0 \le \alpha_n, \alpha_n^* \le C 
\ \text{ for $n = 1, 2, ..., N$}
\end{cases}
\end{aligned}$$

where $\mathbf{G} = \mathbf{\Phi}(\mathbf{X})^T\mathbf{\Phi}(\mathbf{X})$ and $\mathbf{a} = \begin{bmatrix} (\alpha_0 - \alpha_0^*) & (\alpha_1 - \alpha_1^*) & \dots & (\alpha_N - \alpha_N^*) \end{bmatrix}^T$. The matrix $\mathbf{G}$ is called a [Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix). It is a symmetric $\mathbf{N} \times \mathbf{N}$ matrix containing the inner product of every data point $\mathbf{x}$ with every other data point $\mathbf{x}$ in the embedding space. The function that computes this inner product is called a _kernel function_ $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}_{\ge 0}$. Specifically, the entries of $\mathcal{G}$ are given by:

$$ \mathbf{G}_{mn} = 1 + \sum_{i=1}^{D_{emb}} \phi_i(\mathbf{x}_m)\phi_i(\mathbf{x}_n) = 1 + K(\mathbf{x}_m, \mathbf{x}_n)$$ 



## Support Vectors

$$ f(x) = w_0 + \sum_{n=1}^N (\alpha_0 - \alpha_n^*)K(\mathbf{x}_n,\mathbf{x})$$

($w_0 = (\alpha_0 - \alpha_0^*)$)

![Support Vector Regression](support_vectors.svg)

## The Kernel Trick

* Linear Kernel:

$$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T\mathbf{x}'$$

* Polynomial Kernel (degree $d$):

$$K(\mathbf{x}, \mathbf{x}') = (\gamma (\mathbf{x}^T\mathbf{x}') + r)^d$$

* Radial Basis Function (RBF) Kernel:

$$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\lVert \mathbf{x} - \mathbf{x}'\rVert)$$

## Support Vector Regression

(TODO)

## Support Vector Classifiers

(TODO)

## Exercises

:::{dropdown} Exercise 1: Support Vector Regression:
Let's play with some Support Vector Regression (SVR) models. These models are fit by solving the
same maximum absolute error regression problem we discussed above. One benefit of using SVR models
is that it is very easy to try different kinds of embeddings (including infinite-dimensional ones)
simply by swapping out kernels.

Consider the following (randomly generated) 2D dataset:
```
import numpy as np

# Generate training dataset:
data_x = np.array([ 
    np.random.uniform(0,10, 400),
    np.random.uniform(-2e-2,2e-2, 400)
]).T
data_y = np.cos((data_x[:,0]-5)**2/10 + (100*data_x[:,1])**2)
```

Plot the dataset, and fit three different SVR models to the data with [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Use the following kernels:

1. linear kernel (`SVR(kernel='linear')`)
2. second degree polynomial kernel (`SVR(degree=2,kernel='poly')`)
3. radial basis function kernel (`SVR(kernel='rbf')`)

Finally, for each kernel function, plot the prediction surface. You can do this with the following Python function:

```
def plot_model_predictions(data_x, model, x_scaler=None, title=None):
    """
        Plots the prediction surface of a model with 2D features.

        Args:
            data_x: the 2D data (numpy array with shape (N,2))
            model: an sklearn model (fit to normalized `data_x`)
            x_scaler: an sklearn.preprocessing.StandardScaler
                      for normalizing the `data_x` data array
                      (optional)
            title: title of plot (optional)
    """
    mesh_x1, mesh_x2 = np.meshgrid(
        np.linspace(np.min(data_x[:,0]),np.max(data_x[:,0]), 100),
        np.linspace(np.min(data_x[:,1]),np.max(data_x[:,1]), 100)
    )

    mesh_x = np.array([ mesh_x1.flatten(), mesh_x2.flatten() ]).T
    mesh_z = x_scaler.transform(mesh_x) if x_scaler else mesh_x

    pred_y = model.predict(mesh_z)
    mesh_yhat = pred_y.reshape(mesh_x1.shape)

    plt.figure()
    cnt = plt.contourf(mesh_x1, mesh_x2, mesh_yhat, levels=20)
    plt.colorbar(cnt, label=r'$\hat{y}$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title if title else str(model))
    plt.show()
```
:::


:::{dropdown} Exercise 2: Kernel Support Vector Classification

Repeat Exercise 1, but using Support Vector Classifier models (see [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)). Use the SVCs to fit the following training data:

```
import numpy as np

# Generate training dataset:
data_x = np.array([ 
    np.random.uniform(0,10, 400),
    np.random.uniform(-2e-2,2e-2, 400)
]).T
data_y = np.cos((data_x[:,0]-5)**2/10 + (100*data_x[:,1])**2)
```

Above, `data_y` contains +1 values for the positive class and -1 values for the negative class. Use the same three kernels as in Exercise 1:

1. linear kernel (`SVC(kernel='linear')`)
2. second degree polynomial kernel (`SVC(degree=2,kernel='poly')`)
3. radial basis function kernel (`SVC(kernel='rbf')`)

To plot the prediction surfaces of the classifier, you can use the same `plot_model_predictions` function as in Exercise 1.

:::

### Solutions

#### Exercise 1: Support Vector Regression

```{code-cell}
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Generate training dataset:
data_x = np.array([ 
    np.random.uniform(0,10, 400),
    np.random.uniform(-2e-2,2e-2, 400) 
]).T
data_y = np.cos((data_x[:,0]-5)**2/10 + (100*data_x[:,1])**2)

# plot training dataset:
plt.figure()
sp = plt.scatter(data_x[:,0], data_x[:,1], c=data_y, label='Dataset')
plt.colorbar(sp, label=r'$y$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()

def plot_model_predictions(data_x, model, x_scaler=None, title=None):
    """
        Plots the prediction surface of a model with 2D features.

        Args:
            data_x: the 2D data (numpy array with shape (N,2))
            model: an sklearn model (fit to normalized `data_x`) 
            x_scaler: an sklearn.preprocessing.StandardScaler
                      for normalizing the `data_x` data array
                      (optional)
            title: title of plot (optional)
    """
    mesh_x1, mesh_x2 = np.meshgrid(
        np.linspace(np.min(data_x[:,0]),np.max(data_x[:,0]), 100),
        np.linspace(np.min(data_x[:,1]),np.max(data_x[:,1]), 100)
    )

    mesh_x = np.array([ mesh_x1.flatten(), mesh_x2.flatten() ]).T
    mesh_z = x_scaler.transform(mesh_x) if x_scaler else mesh_x
    
    pred_y = model.predict(mesh_z)
    mesh_yhat = pred_y.reshape(mesh_x1.shape)
    
    plt.figure()
    cnt = plt.contourf(mesh_x1, mesh_x2, mesh_yhat, levels=20)
    plt.colorbar(cnt, label=r'$\hat{y}$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title if title else str(model))
    plt.show()

# fit a StandardScaler to data:
scaler = StandardScaler()
scaler.fit(data_x)

# normalize x_data:
data_z = scaler.transform(data_x)

# These are the regression models we are trying:
svr_models = [
    SVR(kernel='linear'),
    SVR(kernel='poly', degree=2),
    SVR(kernel='rbf'),
]

# fit each model and plot the prediction surface:
for svr_model in svr_models:
    
    # fit model to normalized data:
    svr_model.fit(data_z, data_y)
    
    # plot predictions made by kernel SVR model
    plot_model_predictions(data_x, 
                           model=svr_model, 
                           x_scaler=scaler,
                           title=f'SVR ({svr_model.kernel} kernel)')
    
     
```

#### Exercise 2: Kernel Support Vector Classification

```{code-cell}
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# !! Note: This uses the plot_model_predictions function from Exercise 1.


# Generate training dataset:
data_x = np.array([ 
    np.random.uniform(0,10, 400),
    np.random.uniform(-3e-2,3e-2, 400)
]).T
data_y = np.sign(3-(data_x[:,0]**(0.7) + (50*data_x[:,1])**2))

# plot training dataset:
plt.figure()
plt.scatter(data_x[data_y > 0,0], data_x[data_y > 0,1], label='Positive Class')
plt.scatter(data_x[data_y < 0,0], data_x[data_y < 0,1], label='Negative Class')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()

# fit a StandardScaler to data:
scaler = StandardScaler()
scaler.fit(data_x)

# normalize x_data:
data_z = scaler.transform(data_x)

# These are the classifier models we are trying:
svc_models = [
    SVC(kernel='linear'),
    SVC(kernel='poly', degree=2),
    SVC(kernel='rbf'),
]

# fit each model and plot the prediction surface:
for svc_model in svc_models:
    
    # fit model to normalized data:
    svc_model.fit(data_z, data_y)
    
    # plot predictions made by kernel SVR model
    plot_model_predictions(data_x,
                           model=svc_model,
                           x_scaler=scaler,
                           title=f'SVC ({svc_model.kernel} kernel)')
```


