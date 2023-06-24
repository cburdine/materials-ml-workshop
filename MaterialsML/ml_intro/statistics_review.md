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


# Statistics Review

Before we dive into Machine Learning, we will do a brief review of the following concepts from statistics:

* Probability Distributions
* The Binomial Distribution
* The Normal Distribution
* The Central Limit Theorem
* Hypothesis Testing
* The Multivariate Normal Distribution

If you are already familiar with these concepts, feel free to skip this section or to only read the sections you need to review.

## Probability Distributions:

A _random variable_ $X$ is a variable that can take on one of a number of possible values with associated probabilities. The set of possible values attainable by $X$ is called the _support_ of $X$. In this workshop, we will use the notation $\mathcal{X}$ to denote the support of a random variable $X$.

Random variables are often defined as _probability distributions_ over their respective supports. A probability distribution is a function that assigns a likelihood to each possible value $x$ in the support $\mathcal{X}$. Probability distributions can be _discrete_ (i.e. when $\mathcal{X}$ is countable) or _continuous_ (when the $\mathcal{X}$ is not countable). For example, the probability distribution of outcomes for rolling a six-sided dice is discrete, whereas the distribution of darts thrown at a dartboard is continuous. In this workshop, we will use the notation $p(x)$ to denote probability distributions.


In order for a probability distribution to be well-defined, we require the distribution to be _normalized_, meaning all probabilities add up to 1. This means that:

$$1 = \begin{cases} \sum_{x} p(x) &  [\text{for discrete } p(x)]\\ \int_\mathcal{X} p(x)\ dx & [\text{for continuous } p(x)] \end{cases}$$

The _expected value_ of a distribution $p(x)$, denoted $\mathbb{E}[x]$ is given by:

$$\mathbb{E}[p(x)] = \begin{cases} \sum_{x} p(x)x &   [\text{for discrete } p(x)]\quad \\ \int_{\mathcal{X}} p(x)x\ dx  & [\text{for continuous } p(x)] \end{cases}$$

The _expected value_ of a random variable, sometimes called the _average_ value or _mean_ value, is the average of all possible outcomes weighted according to their likelihoods. The mean of a random variable is also often denoted by $\mu$.

:::{note}
In physics and quantum chemistry, you might encounter _Dirac notation_, which uses the notation $\langle x \rangle$ to denote the expected value of $x$. Often, this is referred to as the "expectation value", instead of the "expected value".
:::

The _variance_ of a random variable $X$ describes the degree to which the distribution deviates from the mean $\mu$. It is often denoted by $\sigma^2$, and is given by:

$$\sigma^2 = \mathbb{E}[ (X - \mu)^2 ] = \sum_{x} (x - \mu)^2 = \int_\mathcal{X} (x - \mu)^2\ dx$$

The variance can also be computed by the equivalent formula:

$$\sigma^2 = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

The _standard deviation_ of a distribution, denoted by $\sigma$, is the square root of the variance $\sigma$. Roughly speaking, $\sigma$ measures how far we expect a random variable to deviate from its mean. As a general rule of thumb, if an outcome is more than $2\sigma$ away from $\mu$, it is considered to be a statistically significant deviation.

## The Binomial Distribution:

The _binomial distribution_ is a discrete probability distribution that models the number of successes in a set of $N$ independent trials, where each trial succeeds with a fixed probability $p$. A random variable $X$ that is binomially distributed has support $\mathcal{X} = \{ 0, 1, ..., N \}$ and probability distribution:

$$p(x) = p^{x} (1-p)^{N-x} \binom{N}{x} = p^x (1-p)^{N-x} \left[ \frac{N!}{x!(N-x)!} \right]$$

:::{Note}

We emphasize that $p(x)$ is not the same as $p$. $p$ is the probability of success within any single, intedpendent trial (experiment), so that $(1-p)$ is the probability of failure in any trial. We interpret $p(x)$ as the probability that in a set of $N$ trials, exactly $x$ trials are successful, and $N-x$ trials are failures.
:::

Let's write some Python code to visualize a Binomial distribution. We can compute the probability distribution by hand, or we can use the [`scipy.stats.binom.pmf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html) function:


```{code-cell}
:tags: [hide-input]
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

N = 10  # number of trials
p = 0.3 # probability of trial success

# evaluate probability distribution:
x_support = np.arange(N+1)
x_probs = binom.pmf(x_support, n=N, p=p)

# plot distribution:
plt.figure()
plt.xticks(x_support)
plt.bar(x_support, x_probs)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.show()
```

The mean and variance of this distribution are $\mu = Np$ and $\sigma^2 = np(1-p)$ respectively.

## The Normal Distribution:

The _normal distribution_ (also called the _Gaussian distribution_) is perhaps the most important continuous distribution in statistics. This distribution is parameterized by its mean $\mu$ and standard deviation $\sigma$ and has support $\mathcal{X} = (-\infty, \infty)$. The distribution is:

$$p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{1}{2}\left[\frac{x - \mu}{\sigma}\right]^2\right)$$

If we plot this distribution (using [`scipy.stats.norm.pdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy-stats-norm)), we obtain the familiar "bell curve" shape:

```{code-cell}
:tags: [hide-input]
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

mu = 0.0 # mean of distribution
sigma = 1.0 # standard deviation of distribution

# evaluate probability distribution:
x_pts = np.linspace(mu-3*sigma, mu+3*sigma, 1000)
x_probs = norm.pdf(x_pts, loc=mu, scale=sigma)

# plot distribution:
plt.figure()
plt.fill_between(x_pts, x_probs)
plt.xlabel('x')
plt.ylabel('p(y)')
plt.show()
```


## The Central Limit Theorem

The _[Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)_ and the _[Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)_ are two important theorems in statistics. The Law of Large Numbers states states that as the number of samples $n$ of a random variable $X$ increases, the average of these samples aproaches the distribution mean $\mu = \mathbb{E}[X]$:

$$\text{For samples } x_1, x_2, ..., x_n,\quad \sum_{i=1}^n \frac{x_i}{n} \rightarrow \mu \quad\text{ as }\quad n \rightarrow \infty.$$

The Central Limit Theorem generalizes the Law of large numbers. It states that for a set of $n$ independent samples $x_1, x_2, ..., x_n$ from _any_ random variable $X$ with bounded mean $\mu_X$ and variance $\sigma_X$, the sample mean random variable $\bar{X}_n \sim \sum_{i=1}^n x_i/n$ is such that:

$$\sqrt{n}(\bar{X}_n - \mu_X)\ \underset{distribution}{\longrightarrow}\ \text{Normal}(\mu=0, \sigma=\sigma_X)$$

This theorem is useful for quantifying the uncertainty of the sample mean. If we divide both sides by $\sqrt{n}$ and shift by $\mu_X$, we see that:

$$\bar{X}_n \sim \text{Normal}(\mu=\mu_X,\sigma=\sigma_X/\sqrt{N})$$

In other words, the standard deviation of the sample mean $\bar{x} = \sum_{i=1}^n x_i/n$ is roughly $\sigma_X/\sqrt{n}$. This relation quantifies the uncertainty of using the sample mean as an estimate of a population mean.


## Hypothesis Testing

An important part of doing science is the testing of hypotheses. The standard way of doing this is through the steps of the _scientific method_: Formulate a research question, propose a hypothesis, design an experiment, collect experimental data, analyze the results, and report conclusions. In the analysis of our data, how do we know if our hypothesis is correct? There are many different statistical methods we can apply to test a given hypothesis, each with different strengths and weaknesses. In machine learning, we often use hypothesis testing to determine (hopefully with a high degree of certainty) whether one model is more accurate than another. We can also use hypothesis testing to determine which data features are more significant than other data features when making predictions.

Typically, hypothesis testing involves two competing hypotheses: the _null hypothesis_ (denoted $H_0$) and the _alternative hypothesis_ (denoted $H_1$). The null hypothesis often is a statement of the "status quo" or a statement of "statistical insignificance". The alternative hypothesis is the statement of "statistical significance" we are often trying to prove is true. To better illustrate the process of hypothesis testing, we will use the following example:


### Example: Conductor vs. Insulator Classifier

Suppose we are developing a classifier model that predicts whether a material is a conductor or an insulator. For simplicity, we shall assume that roughly half of all materials are insulators and half are insulators. Our two competing hypotheses would then be:

* $H_0$: The accuracy of our classifier is the same as random guessing (accuracy = 0.5)
* $H_1$: The accuracy of our classifier is better than than random guessing (accuracy > 0.5)

Suppose that in order to test our alternative hypothesis $H_1$, we compile a dataset of 40 materials (20 conductors and 20 insulators) and use these to evaluate our model. We find that the model has an accuracy of 0.6, meaning it correctly classifies $60\%$ of the dataset. Since the accuracy is greater than 0.5, does this mean we immediately reject $H_0$ in favor of $H_1$? Not necessarily; it could be the case that our model simply got lucky and "randomly guessed" the classification of more than $50\%$ of the dataset.

First, let's consider the distribution of accuracies that could be attained by a random guessing strategy. If we treat each guess as one of $N = 40$ trials with a probability $p = 0.5$ of succeeding, we can model the distribution of random guessing strategies with a binomial distribution. Let's write some Python code to visualize this distribution:

```{code-cell}
:tags: [hide-input]
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# experiment parameters:
N = 40
p_guess_correct = 0.5
model_accuracy = 0.6

# evaluate distribution:
n_correct = np.array(np.arange(N+1))
probs = binom.pmf(n_correct,n=N,p=p_guess_correct)
accuracies = n_correct / N

# plot distribution
plt.figure()
plt.bar(accuracies, probs, width=1/N)
plt.axvline(x=model_accuracy, color='r', ls=':', label='Model Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('P(Accuracy)')
plt.legend()
plt.show()
```

In order to evaluate whether or not our result is statistically significant, we will compute the [_$p$-value_](https://en.wikipedia.org/wiki/P-value) associated with our hypothesis testing. A $p$-value is a quantity between $0$ and $1$ that describes the probability of obtaining a result at least as exteme as the experimentally observed value assuming that $H_0$ is true. Roughly speaking, we can interpret a $p$-value as the probability of observing the experimental data "by coincidence" if $H_0$ is in fact true. If a $p$-value is low, it means that the alternative hypothesis $H_1$ is likely to be true. In most research settings, a p-value of at most $0.05$ ($5\%$ chance of coincidence) is considered sufficient to show that the alternative hypothesis $H_1$ is true.


From inspecting thie plot of the distribution above, we see that the accuracy distribution is approximately normal, having mean $\mu_X \approx p = 0.5$ and variance $\sigma^2_X \approx p(1-p) = 0.25$. Per the Central Limit Theorem, we conclude that the estimated accuracy of random guessing is normally distributed with mean $\mu = \mu_X$ and $\sigma = \sigma_X/\sqrt{40}$. The $p$-value corresponds to the area under this normal distribution curve corresponding to accuracies with $0.6$ or greater. Using the values from the previous code cell, we can compute the $p$-value as follows:

```{code-cell}
:tags: [hide-input]
from scipy.stats import norm

# plot approximate normal distribution via CLT:
mu_clt = p_guess_correct
sigma_clt = p_guess_correct*(1-p_guess_correct)/np.sqrt(N)

# evaluate CLT normal distribution:
x_pts = np.linspace(mu_clt-4*sigma_clt, mu_clt+4*sigma_clt, 1000)
x_probs = norm.pdf(x_pts, loc=mu_clt, scale=sigma_clt)

# determine region to the right of estimated accuracy:
pval_pts = x_pts[x_pts > model_accuracy]
pval_probs = x_probs[x_pts > model_accuracy]

# estimate pvalue:
pvalue = 1 - norm.cdf(model_accuracy, loc=mu_clt, scale=sigma_clt)

# plot CLT normal distribution and shade region
# to the right of estimated accuracy:
plt.figure()
plt.plot(x_pts, x_probs, 'k')
plt.fill_between(pval_pts, pval_probs, label=f'p-value: {pvalue:.3f}')
plt.axvline(x=model_accuracy, color='r', ls=':', label='Model Accuracy')
plt.legend()
plt.show()
```

Since the $p$-value is $0.006 \le 0.05$, we conclude that the $H_1$ is true, meaning the accuracy of our model ($0.6$) being greater than random guessing ($0.5$) is statistically significant. This proves that the model is better than random guessing; however it is worth noting that a model with an accuracy of $0.6$ may not be practically useful for distinguishing between insulators and metals.

## The Multivariate Normal Distribution:

Often, we will find that we are working with multi-dimensional data where correlations may exist between more than one variable. Fortunately, these correlations can be described by a [multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) distribution. Like the 1-dimensional normal distribution, the multivariate normal distribution is characterized by two parameters, a mean vector ${\boldsymbol{\mu}}$ and a [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) $\mathbf{\Sigma}$. For a $d$-dimensional distribution, these parameters can be written in matrix form:

$$\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_d \end{bmatrix}, \qquad\qquad \mathbf{\Sigma} = \begin{bmatrix}
\sigma_{1}^2 & \sigma_{12} & \dots & \sigma_{1d} \\
\sigma_{21} & \sigma_{2}^2 & \dots & \sigma_{2d} \\
\vdots      & \vdots      & \ddots & \vdots \\
\sigma_{d1} & \sigma_{d2} & \dots & \sigma_{d}^2
\end{bmatrix}$$

The entries $\mu_i = \mathbb{E}[X_i]$ are the coordinates of the mean $\boldsymbol{\mu}$. The entries $\sigma_i^2 = \mathbb{E}[(X_i - \mu_i)^2]$ in $\Sigma$ are the variances of each individual component of the distribution. Finally, the off-diagonal components $\sigma_{ij}$ are the _covariances_ of components $i$ and $j$. The covariance of two components is given by:

$$\text{Cov}(X_i,X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)] = \iint_{\mathcal{X}_i \times \mathcal{X_j}} p(x_i,j_j)(x_i - \mu_i)(x_j - \mu_j)\ dx_jdx_i$$

The probability distribution of a multivariate normal distribution is given by:

$$p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

:::{note}
From the definition of $\text{Cov}(X_i, X_j)$, it follows that $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$. This means that the covariance matrix $\mathbf{\Sigma}$ is symmetric ($\mathbf{\Sigma} = \mathbf{\Sigma}^T$), having up to $d(d+1)/2$ distinct values that need to be determined.

For any two random variables, if $\text{Cov}(A,B) = 0$ the random variables are uncorrelated; otherwise, the sign of $\text{Cov}(A,B)$ indicates whether $A$ and $B$ are positively or negatively correlated.

Also, for the multivariate normal distribution to be well-defined, we must impose that the matrix $\mathbf{\Sigma}$ is invertible. If $\mathbf{\Sigma}$ is not invertible, $\det(\mathbf{\Sigma}) = 0$, which means $p(\mathbf{x})$ cannot be normalized.
:::

To evaluate the density of a multivariate normal distribution, we can use the [`scipy.stats.multivariate_normal.pdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html) function:

```{code-cell}
:tags: [hide-input]
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# mean of distribution:
mu = np.array([ 0.0, 0.0 ])

# covariance matrix of distribution:
sigma = np.array([
    [  1.0, -1.0 ],
    [ -1.0,  2.0 ]
])

# define 2D mesh grid:
x1_pts = np.linspace(-4,4,100)
x2_pts = np.linspace(-4,4,100)
x1_mesh, x2_mesh = np.meshgrid(x1_pts, x2_pts)
x12_mesh = np.dstack((x1_mesh, x2_mesh))

# evaluate probability density on mesh points:
prob_mesh = multivariate_normal.pdf(x12_mesh, mean=mu, cov=sigma)

# plot distribution:
plt.figure()
plt.contourf(x1_mesh, x2_mesh, prob_mesh, levels=10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.colorbar(label='p(x)')
plt.show()
```
## Exercises

:::{dropdown} Exercise 1: Comparing Two Classifiers:

Consider the Conductor vs. Insulator Classifier example from above. Suppose that after some additional work, we propose an improved classifier model, which we think can classify materials with an accuracy greater than $0.6$. We now have the following two Hypotheses:

* $H_0$: The accuracy of the improved classifier is the same as the original classifier (accuracy = 0.6)
* $H_1$: The accuracy of the improved classifier is the greater than the original classifier (accuracy > 0.6)

Suppose we evaluate the accuracy of the improved classifier with a dataset of only 40 materials, and find that $26$ of the $40$ materials are classified correctly. Repeat the same analysis as before to determine whether $H_1$ is true and report the $p$-value.

Next, suppose that we instead used a dataset of $80$ materials and found that $52$ of these were classified correctly (same accuracy as before, but more data). Does this change our conclusion as to whether or not $H_1$ is true?
:::

:::{dropdown} Exercise 2: Fitting a Multivariate Normal Distribution:

Generate a 2D dataset of 10,000 random points uniformly sampled within a rectangle of width $2$ and height $1$, where the lower lefthand corner of the rectangle is fixed at the origin. You can generate uniform values on the interval $[a,b]$ using the [`np.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html) function in the numpy package.

Fit this distribution to a multivariate normal distribution by computing the sample mean vector $\boldsymbol{\mu}$ and sample covariance matrix $\mathbf{\Sigma}$. You can do this with the [`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) function and the [`np.cov`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) functions respectively (these are also contained in the `numpy` package).

Plot both the generated data points and the fitted multivariate normal distribution using [`plt.contourf`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html) or a similar function.
:::

### Solutions:

#### Exercise 1: Comparing Two Classifiers
```{code-cell}
:tags: [hide-cell]
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Original model accuracy:
original_accuracy = 0.6

# Experimental results:
# (number correct, total)
experiment_results = [
    (26,40), (52,80)
]

# perform analysis for N = 40 and 80:
for (n_correct, N) in experiment_results:

    # compute accuracy:
    accuracy = n_correct/N

    # compute CLT mu and sigma:
    mu_clt = original_accuracy
    sigma_clt = original_accuracy*(1-original_accuracy)/np.sqrt(N)

    # evaluate CLT normal distribution:
    x_pts = np.linspace(mu_clt-4*sigma_clt, mu_clt+4*sigma_clt, 1000)
    x_probs = norm.pdf(x_pts, loc=mu_clt, scale=sigma_clt)

    # determine region to the right of estimated accuracy:
    pval_pts = x_pts[x_pts > accuracy]
    pval_probs = x_probs[x_pts > accuracy]

    # estimate pvalue:
    pvalue = 1 - norm.cdf(accuracy, loc=mu_clt, scale=sigma_clt)

    # plot CLT normal distribution and shade region
    # to the right of estimated accuracy:
    plt.figure(figsize=(6,3))
    plt.title(f'N = {N} Dataset')
    plt.plot(x_pts, x_probs, 'k')
    plt.fill_between(pval_pts, pval_probs, label=f'p-value: {pvalue:.3f}')
    plt.axvline(x=accuracy, color='r', ls=':', label='Model Accuracy')
    plt.legend()
    plt.show()

    # print out conclusion:
    if pvalue > 0.05:
        print('pvalue > 0.05, so we reject H0')
    else:
        print('pvalue < 0.05, so we accept H0')
```

#### Exercise 2: Fitting a Multivariate Normal Distribution:

```{code-cell}
:tags: [hide-cell]
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# randomly sample the x1 and x2 coordinates of points:
n_data  = 1000
x1_data = np.random.uniform(0,2,n_data)
x2_data = np.random.uniform(0,1,n_data)

# combine x1 and x2 components:
x_data = np.array([ x1_data, x2_data ])
x_mean = np.mean(x_data, axis=-1)
x_cov = np.cov(x_data)

# define 2D mesh grid:
x1_pts = np.linspace(-0.5,2.5,100)
x2_pts = np.linspace(-0.5,1.5,100)
x1_mesh, x2_mesh = np.meshgrid(x1_pts, x2_pts)
x12_mesh = np.dstack([x1_mesh, x2_mesh])

# evaluate probability density on mesh points:
prob_mesh = multivariate_normal.pdf(x12_mesh, mean=x_mean, cov=x_cov)

# show plot of fitted multivariate normal distribution:
plt.figure()
plt.contourf(x1_mesh, x2_mesh, prob_mesh, levels=10)
plt.scatter(x1_data, x2_data, color='r', s=1)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
```
