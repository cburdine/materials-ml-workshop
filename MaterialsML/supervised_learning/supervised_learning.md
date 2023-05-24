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

# Supervised Learning

Now that we have reviewed some of the necessary background material, we will will begin examining the most common type of machine learning problem: _supervised learning_. Supervised learning is applied to problems where the available data contains many different labeled examples, and the problem involves finding a model that maps a set of features (inputs) to labels (outputs).

Although data can take many different forms (i.e. numbers, vectors, images, text, 3D structures, etc.), we can think of a dataset as a collection of of $(\mathbf{x}, y)$ pairs, where $\mathbf{x}$ is a set of features, and $y$ is the label to be predicted. For now, we will consider the simplest case where $\mathbf{x}$ is a vector of floating-point numbers and $y$ is either (a) one of a finite number of mutually classes, or (b) a scalar quantity. In case (a), the supervised learning problem is a _classification problem_, where we must learn a model that makes prediction $\hat{y}$ of the class $y$ associated with the set of features $\mathbf{x}$. In case (b), the supervised learning problem is a _regression problem_, where we must learn a model that produces an estimate $\hat{y}$ of $y$ based on $\mathbf{x}$.

For both classification and regression problems, we can think of a model as a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that maps the space of possible features $\mathcal{X}$ into the space of possible labels $\mathcal{Y}$.

![Illustration of a Supervised Model](supervised_model.svg)

Ideally, we would like the function $f$ to be a _valid_ model, meaning it maps every possible set of features in $\mathcal{X}$ to the correct output label; however, finding such a function may be impossible for a number of reasons. First, it is possible that $\mathcal{X}$ might be an infinite set, meaning it is impossible to verify that $f$ is valid for all sets of features $\mathbf{x}$. In some situations, we may not even know what the correct labels are for every single $\mathbf{x}$. For these reasons, it might appear that _learning_ a valid model $f: \mathcal{X} \rightarrow \mathcal{Y}$ is an impossible challenge.  

Fortunately, we have a powerful tool that we can use to tackle this seemingly insurmountable task: _data_. Using the data we have gathered, we can estimate the validity of a model by determining how well it _fits_ the data. Quantitatively speaking, we say that a model $f$ fits a dataset of $(\mathbf{x}_i,y_i)$ pairs if $f(\mathbf{x}_i) \approx y_i$ for each $i$. 

We also know that following statement is generally true with regards to our data and any potential model $f$:

> If a model $f : \mathcal{X} \rightarrow \mathcal{Y}$ is valid, then it will fit the data.

Indeed, any model that is valid should produce a good fit to the data, provided that enough features are accounted for in $\mathcal{X}$ and that sufficient care is taken to eliminate noise in the data.

Now, take a moment to consider the converse of the previous statement:

> If a model $f : \mathcal{X} \rightarrow \mathcal{Y}$ fits the data, then it is valid.

Is this statement also true? At first glance, you might be tempted into thinking it might be, but it is in general not true. In fact, sometimes a model that fits the data better than another model may actually be less valid. An an illustration of this, consider the following regression problem, where we have proposed two different fits:

```{code-cell}
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

# define data distribution:
def y_distribution(x):
	return -0.1*x + 0.9 + \
            np.random.uniform(-0.3,0.3,size=len(x))

# the data we obtain will naturally fit the valid model with
# some noise due to experimental errors:
data_n = 12
data_x = np.linspace(0,10,data_n)
data_y = y_distribution(data_x)

# fite data to a line (degree 1 polynomial):
xy_linefit = np.poly1d(np.polyfit(data_x,data_y,deg=1))

# fit data to N-1 degree polynomial to data:
xy_polyfit = np.poly1d(np.polyfit(data_x,data_y,deg=len(data_x)-1))

# plot data and valid/invalid models for comparison:
plt.figure()
eval_x = np.linspace(0,10,1000)
plt.scatter(data_x,data_y, color='g', label=r'Data $(x,y)$')
plt.plot(eval_x, xy_linefit(eval_x), label=r'Linear Fit $f(x)$')
plt.plot(eval_x, xy_polyfit(eval_x), label=r'Polynomial Fit $f(x)$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

:::{important}
Since the data above is randomly generated, notebook output may vary.
:::

Above, we have randomly generated data and proposed two models: a _linear fit_ of the form $f(x) = mx + b$ and a degree 11 polynomial fit of the form $f(x) = \sum_{n=0}^{11} a_n x^n$. Since a degree $N-1$ polynomial can perfectly fit $N$ data points, the polynomial fit exactly matches the data, whereas the linear fit does not. Nonetheless, it is likely that a linear fit is the more valid model, especially when we consider that the polynomial fit extrapolates very poorly outside of the interval $[0,10]$ that contained the dataset.

```{code-cell}
:tags: [hide-input]

# determine the region of valid fits:
extrap_x = np.linspace(-2, 12, 1000)
valid_fit_region = [
    -0.1*extrap_x + 0.9 - 0.3,
    -0.1*extrap_x + 0.9 + 0.3
]

# plot how well the fits extrapolate from the data:
plt.figure()
plt.fill_between(extrap_x, valid_fit_region[0], valid_fit_region[1],
                 color='g', alpha=0.1, label='Valid Fit Region')
plt.plot(extrap_x, xy_linefit(extrap_x), label=r'Linear Fit $f(x)$')
plt.plot(extrap_x, xy_polyfit(extrap_x), label=r'Polynomial Fit $f(x)$')
plt.xlim((-2,12))
plt.ylim((-5,5))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

From this exercise, we see that some models are not valid, even though they may perfectly fit the data. So if we have a model that does seem to fit the data set well, how can we be sure that the model is also valid? Unfortunately, we can never know the degree to which a model is valid unless our dataset contains every possible input in $\mathcal{X}$ and its associated label. However, there are some general tactics we can employ to maximize the validity of our model. These are the three most important guidelines to keep in mind when working with supervised models:

* **Be cautious in how the data is obtained.** 

Is the data accurate? Are sources of noise identified and accounted for? Does the dataset adequately span the set of relevant input features $\mathcal{X}$ and labels $\mathcal{Y}$? You can _never_ have too much data, but you _can_ have too much data of a particular kind.

* **Be cautions in how the data is handled.**

How is the data handled to fit the model? Are training, validation, and test sets being used, and are they kept independent of one another? Are you avoiding selection bias? If you are enriching the dataset with additional features, are those features necessary and accurate?

* **Be sure that an appropriate model is being used.**

Does the size of the dataset warrant the complexity of the model you are using? Are any symmetries of the data reflected in the symmetries of the model?


## Obtaing Data

Depending on the problem you are investigating, you may or may not play a role in the data collection process. Since we are materials scientists, the data we are working with often comes from either laboratory measurements or computational simulations. If you happen to have some control over the data collection processs, make every effort to ensure that the data is collected in a consistent manner, and that meaningful features and labels are reported accurately for each laboratory sample. If you are generating data through computational simulations, be sure that any data produced is at least consistent with values measured in the laboratory or with other independently reported values in the literature. After all, your aim is to develop a model of the physics and chemistry of the real world, not a model of the (often simplified) physics and chemistry of a simulation. As a famous computational scientist once said:

```{epigraph}
"The purpose of computing is insight, not numbers."

-- Richard Hamming
```

If you are collecting data for a classification problem, make sure that all classes are represented in a balanced ratio, if possible. This will help a model avoid bias toward predicting the classes that appear more frequently in the dataset. Likewise, for regression problems be sure that the extremes of both $\mathcal{X}$ and $\mathcal{Y}$ are contained in the dataset. This will help mitigate extrapolation error, as we saw in the previous example with the polynomial fit.


# Handling Data

Let's return to the problem of determining model validity. We said earlier that a model $f$ is _valid_ if $f(\mathbf{x}) = \hat{y} \approx y$ for points $\mathbf{x}$ both _inside_ and _outside_ of the dataset. However, we do not know what the correct values in $\mathcal{Y}$ that correspond to feature sets $\mathbf{x}'$ lying outside the dataset: 

![Out-of-distribution Validity](supervised_model_ood.svg)

Without data that is kept separate from the data that the model was fit to, we have no way of estimating the validity of the model, that is, how accurate the model is on $\mathcal{X}$ as a whole, not just on the dataset used to fit the model. This is why it is customary to set aside a subset of the data that is not used for fitting the model, but is used for only ensuring the validity of the model. This reserved subset of the dataset is usually referred to as the _validation set_. By measuring the model's accuracy on the  _validation set_, we can effectively estimate how well the model generalizes to data that was not used. When using the validation set, we must be cautious, since this estimate is only valid if the dataset provides sufficient coverage of the set of possible input features $\mathcal{X}$. In other words, any biases present in the dataset will result in biases in any estimates of model accuracy obtained by a validation set.



![Train-Validation-Test Split](supervised_split.svg)

### The Training Set

(TODO)

### The Validation Set

(TODO)

### The Test Set

(TODO)

### Normalizing Data

(TODO)


## Model Selection

```{epigraph}

"Entities must not be multiplied beyond necessity."

-- Occam's Razor
```

## Key Steps of Supervised Learning

(TODO)

## Exercises

(TODO)
