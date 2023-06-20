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
# Application: Classifying Perovskites

Now that we've covered some of the basic theory of supervised learning, let's start applying some basic supervised learning models to a real dataset.

## Perovskite Classification Dataset

Perovskites are materials with a crystal structure of the form  $ABX_3$, where $A$ and $B$ are two positively charged cations and $B$ is a negatively charged anion, usually oxygen ($O$). Ideal perovskites have a cubic structure, although some perovskites may attain more stable configurations with slightly perturbed structural phases, such as orthorombic or tetragonal. Here's an example of [SrTiO$_3$](https://en.wikipedia.org/wiki/Strontium_titanate) (strontium titanate), which is most stable with a cubic structure:

```{image} SrTiO3_cubic.png
:alt: fishy
:class: bg-primary mb-1
:width: 250px
:align: center
```
We will be using the [AB03 Perovskites](https://www-sciencedirect-com.ezproxy.baylor.edu/science/article/pii/S0927025620306820?via%3Dihub) dataset. 
This data was originally used in the paper [_Crystal structure classification in ABO3 perovskites via machine learning_](https://doi-org.ezproxy.baylor.edu/10.1016/j.commatsci.2020.110191) by _Behara et al_.
Download the dataset and put the `perovskites.csv` file in the same directory as your Python notebook.

Once downloaded, you can load the dataset into a `pandas` dataframe using the following Python code:

```{code-cell}
:tags: [hide-input]
import pandas as pd

# load dataset into a pandas DataFrame:
PEROVSKITE_CSV = 'perovskites.csv'
perovskite_df = pd.read_csv(PEROVSKITE_CSV)

# show dataframe in notebook:
display(perovskite_df)
```

## Data Features

For now, we will focus primarily on the prediction of the perovskite structure (the _Lowest Distortion_ column). In the dataset, there are many different features given for each perovskite material. They include:

* Chemical Formula (with A and B elements)
* Valence of A (0-6 or unlisted): $V(A)$
* Valence of B (0-6 or unlisted): $V(B)$
* Radius of A at 12 coordination: $r(A_{XII})$
* Radius of A at 6 coordination: $r(A_{VI})$
* Radius of B at 6 coordination: $r(B_{VI})$
* Electronegativity of A: $EN(A)$
* Electronegativity of B: $EN(B)$
* Bond length of A-O pair $l(A$-$O)$
* Bond Length of B-O pair $l(B$-$O)$
* Eletronegativity difference with radius: $\Delta ENR$
* Goldschmidt tolerance factor: $t_G$
* New tolerance factor: $\tau$
* Octahedral factor: $\mu$

In total, there are 17 distinct factors in the dataframe; however many of these featurescan be computed directly from other features. For example, the [Goldschmidt tolarance factor](https://en.wikipedia.org/wiki/Goldschmidt_tolerance_factor) is a quantity commonly used to evaluate the stability of perovskite structures. It is computed using the equation:

$$t_G = \frac{(r(A) + r(B))}{\sqrt{2}(r(B) - r(\text{O}))}$$

where $r(A)$ and $r(B)$ are the ionic radii of the $A$ and $B$ elements and $r(\text{O})$ is the ionic radius of oxygen.

To make things a bit simpler, let's consider only the following three factors:

* Electronegativity of A: $EN(A)$
* Electronegativity of B: $EN(B)$
* Goldschmidt tolerance factor: $t_G$

We can select these features and convert them into `numpy` arrays using the following code:

```{code-cell}:
:tags: [hide-input]
import numpy as np

# features:
a_electronegativity = np.array(perovskite_df['EN(A)'])
b_electronegativity = np.array(perovskite_df['EN(B)'])
goldschmidt_tolerance = np.array(perovskite_df['tG'])

# combine features into columns of a Nx3 numpy array:
features = np.array([
    a_electronegativity,
    b_electronegativity,
    goldschmidt_tolerance
]).T

print('Shape of features:', features.shape)
```

Let's also take a look at all of the distinct structures that appear in the _Lowest distortion_ column:

```{code-cell}
:tags: [hide-input]
# dataset labels:
structures = np.array(perovskite_df['Lowest distortion'])

# reduce to distrinct values:
distinct_structures = list(set(structures))
print(distinct_structures)
```

It appears that there are four different distinct phases listed in the dataset: `rhombohedral`, `cubic`, `tetragonal`, and `orthorhombic`. There are also some rows in the dataframe that do not have a phase listed (denoted by `-`). Let's see how many of each phase are listed in the dataset:

```{code-cell}:
:tags: [hide-input]

import matplotlib.pyplot as plt

# determine how many examples of each structure are in the dataset:
bar_counts = np.array([ 
    len(structures[structures == structure_type])
    for structure_type in distinct_structures
])

# plot bar plot of counts for each type of structure:
plt.figure(figsize=(6,2))
bar_locations = np.array(range(len(distinct_structures)))
plt.barh(bar_locations, bar_counts)
plt.gca().set_yticks(bar_locations)
plt.gca().set_yticklabels(distinct_structures)
plt.show()
```
As expected, a majority of the perovskites have the ideal cubic structure. 

## Classifying Cubic versus Non-Cubic Structures:

For simplicity, let's first consider the problem of classifying cubic versus non-cubic structures. This is a simple binary classification task that we can solve using the _binary perceptron_ model we learned about previously. Let's start by removing the data entries with unknown structure (`-`) and assigning values of `1` to cubic structures and `-1` to non-cubic structures:

```{code-cell}
:tags: [hide-input]

# determine the indices of unlabeled  data:
clean_data_idxs = (structures != '-')

# remove unlabeled data from features:
clean_features = features[clean_data_idxs]
clean_structures = structures[clean_data_idxs]

# assign binary classifier labels +1/-1 for cubic/non-cubic:
binary_classifier_labels = np.array([
    1 if struct == 'cubic' else -1
    for struct in structures
])

print('Features shape:', clean_features.shape)
print('Binary classifier labels shape:', binary_classifier_labels.shape) 
```

Let's visualize what the cubic versus non-cubic data looks like with respect to electronegativity (A and B):

```{code-cell}
:tags: [hide-input]

# separate out cubic and non-cubic data:
cubic_data = features[binary_classifier_labels == 1]
noncubic_data = features[binary_classifier_labels == -1]

# plot EN(A), EN(B) for cubic and non-cubic data:
plt.figure()
plt.scatter(cubic_data[:,0], cubic_data[:,1], s=3.0, label='Cubic')
plt.scatter(noncubic_data[:,0], noncubic_data[:,1],s=3.0, label='Non-Cubic')
plt.xlabel('EN(A)')
plt.ylabel('EN(B)')
plt.legend()
plt.show()
```
Since the [electronegativity](https://en.wikipedia.org/wiki/Electronegativity) generally increases with the group and decreases with the period of elements on the periodic table, it serves as a good numerical quantity to associate with each element. This is why we observe a grid-like distribution of the data. From glancing at the distribution of cubic versus non-cubic materials, it appears that there is no immediately identifiable trend.

Let' also look at how the cubic and non-cubic structures are distributed with respect to the Goldschmidt tolerance factor $t_G$:

```{code-cell}
:tags: [hide-input]


# define the histogram "bins" used to visualize the t_G distribution:
hist_bins = np.linspace(np.min(goldschmidt_tolerance), 
                        np.max(goldschmidt_tolerance), 51)

# plot histograms of cubic and noncubic data with respect to t_G:
plt.figure()
plt.hist(cubic_data[:,2], bins=hist_bins, alpha=0.4, label='Cubic')
plt.hist(noncubic_data[:,2], bins=hist_bins, alpha=0.4, label='Non-Cubic')
plt.xlabel('Goldschmidt Tolerance $t_G$')
plt.ylabel('Count')
plt.legend()
plt.show()

```

We see that a majority of the cubic structures are distributed within the range of $t_G = 0.7$ to $t_G = 0.9$. According to the literature, perovskites with $t_G$ in the range of $0.9$ to $1.0$ are generally predicted to be in the cubic phase, which appears to be inconsistent with our data. Taking note of this discrepancy, we will proceed with preparing our data. First, we will split the data into training, validation, and test sets. This can be done with the [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function. We will also normalize our data using [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/preprocessing.html):

```{code-cell}
:tags: [hide-input]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# split train and non-training data 80% and 20%:
x_train, x_nontrain, y_train, y_nontrain = train_test_split(
            features, binary_classifier_labels, train_size=0.8)

# further split non-training data into validation and test data:
x_val, x_test, y_val, y_test = train_test_split(
            x_nontrain, y_nontrain, test_size=0.5)

# Determine the normalizing transformation:
x_scaler = StandardScaler()
x_scaler.fit(x_train)

# transform x -> z (normalize x data):
z_train = x_scaler.transform(x_train)
z_val = x_scaler.transform(x_val)
z_test = x_scaler.transform(x_test)

print('Shape of z_train:', z_train.shape)
print('Shape of z_val:', z_val.shape)
print('Shape of z_test:',z_test.shape)
```

## The Perceptron (Linear Classification) Model

One of the simplest binary (i.e. two-class) classification models is a _linear classifier_ model, historically referred to as a [_Perceptron_](https://en.wikipedia.org/wiki/Perceptron) model.

For a feature vector $\mathbf{x}$ with $N$ features, a linear classifier model $f(\mathbf{x})$ makes "0" and "1" class predictions according to the equation:

$$f(\mathbf{x}) = \begin{cases}
1, & w_0 + \sum_{i=1}^N w_ix_i > 0 \\
0, & \text{ otherwise}
\end{cases}$$

```{code-cell}
:tags: [hide-input]

from sklearn.linear_model import Perceptron

# fit a linear perceptron model to training set:
perceptron = Perceptron()
perceptron.fit(z_train, y_train)

# evaluate the model on the training and validation set:
train_accuracy = perceptron.score(z_train, y_train)
val_accuracy = perceptron.score(z_val, y_val)

# print the accuracy of the model:
print('training set accuracy:  ', train_accuracy)
print('validation set accuracy:', val_accuracy)
```



## The Nearest Neighbor Model

```{code-cell}
:tags: [hide-input]

from sklearn.neighbors import KNeighborsClassifier

# fit a k-nearest neighbor classifier (k=3 in this case):
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(z_train, y_train)

# evaluate the model on the training and validation set:
train_accuracy = knc.score(z_train, y_train)
val_accuracy = knc.score(z_val, y_val)

# print the accuracy of the model:
print('training set accuracy:  ', train_accuracy)
print('validation set accuracy:', val_accuracy)
```

```{code-cell}
:tags: [hide-input]

en_range = (0.75, 2.6)
eval_tG = np.mean(x_train[:,2])

mesh_size = 200
a_mesh, b_mesh = np.meshgrid(
    np.linspace(en_range[0],en_range[1], mesh_size),
    np.linspace(en_range[0],en_range[1], mesh_size))

mesh_features = np.array([
    a_mesh.flatten(),
    b_mesh.flatten(),
    eval_tG*np.ones_like(a_mesh.flatten())
]).T

z_mesh_features = x_scaler.transform(mesh_features)

predictions = knc.predict(z_mesh_features)
predictions_mesh = predictions.reshape(a_mesh.shape)

plt.figure()
surf = plt.contourf(a_mesh, b_mesh, predictions_mesh, cmap='binary', levels=100)
cbar = plt.colorbar(surf)
cbar.ax.set_yticks([-1,1])
cbar.ax.set_yticklabels(['Cubic','Non-Cubic'])
plt.xlabel('EN(A)')
plt.ylabel('EN(B)')
plt.title(r'Nearest Neighbor Classification [$t_G$ = ' + f'{eval_tG:.3f}]')
plt.show()
```



## Exercises

### Solutions

