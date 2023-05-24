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
# Basic Supervised Models

Now that we've covered some of the basic theory of supervised learning, let's start applying some basic supervised learning models to a real dataset.

## Example Dataset: Perovskite Classification

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

# combine features into a 3xN numpy array:
features = np.array([
    a_electronegativity,
    b_electronegativity,
    goldschmidt_tolerance
])

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

## Nearest-Neighbor Models

### Classification



### Regression


## Exercises

### Solutions

