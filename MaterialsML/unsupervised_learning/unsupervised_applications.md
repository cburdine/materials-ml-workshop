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

# Application: Classifying Superconductors

Now that we have a strong understanding of unsupervised learning, let's apply what we have learned to study a class of materials known as _superconductors_. Superconductors are materials that exhibit zero electrical resistance below a critical temperature $T_c$. Understanding patterns in their compositions and external conditions (such as pressure applied to the material) can help identify trends that correlate with desired high-$T_c$ behavior. These insights could be leveraged to discover new kinds of superconductors or to help engineer the external conditions to raise the $T_c$ of known superconductors.

You can download the dataset for this section using the following Python code:

```
import requests

CSV_URL = 'https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/unsupervised_learning/matml_supercon_cleaned.csv'

r = requests.get(CSV_URL)
with open('matml_supercon_cleaned.csv', 'w', encoding='utf-8') as f:
    f.write(r.text)
```

Alternatively, you can download the CSV file directly [here](https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/unsupervised_learning/matml_supercon_cleaned.csv).

## Loading the Dataset

Let's begin by loading the dataset in to a Pandas dataframe and then perform some basic data cleaning.
Specifically, we will remove entries with missing compositions and those with anomalously high $T_c$ values (likely errors or non-physical outliers). We then display the dataset to view the data features:

```{code-cell}
:tags: [hide-input]
import pandas as pd

# load dataset into a pandas DataFrame:
SUPERCON_CSV = 'matml_supercon_cleaned.csv'
data_df = pd.read_csv(SUPERCON_CSV)

# remove rows for materials with an ambiguous composition:
data_df = data_df[data_df['Composition'].isnull() == False]

# remove corrupted data with invalid Tc:
data_df = data_df[data_df['Tc (K)'] < 400]

# show dataframe in notebook:
display(data_df)
```

Here's a summary of the different features included in the dataset:

* _Material_: The chemical formula of the superconducting material.
* _Substitutions_: Python dictionary consisting of doping ratios.
* _Composition_: Composition of the chemical formula under doping.
* _Tc_: Reported critical temperature of superconductivity (K).
* _Pressure_: Applied pressure to superconducting sample (GPa).
* _Shape_: Shape of the superconducting sample (if any).
* _Substrate_: Substrate the sample was deposited on (if any).
* _DOI_: DOI of paper that data was extracted from.

In this section, we will not attempt to predict any superconductor properties; instead, we will aim to extract insight from this dataset by applying clustering and distribution estimation models.

## Preprocessing Data

Machine learning models require numerical inputs, so we will need to convert the materials' compositions (stored as Python dictionaries in string form) into fixed-length feature vectors. First, we will normalize the elemental compositions to account for different formula unit sizes. We will also append the applied pressure and $T_c$ to each feature vector. Finally, we will standardize the data to have zero mean and unit variance—a common preprocessing step for most ML algorithms.


To simplify the data conversion process, we will create a set `ELEMENTS` that contains all elements
appearing in the dataset. We will also create a function called `vectorize_composition()` that will convert each chemical composition dictionary into a minimal feature vector:

```{code-cell}
:tags: [hide-input]
import numpy as np

# Generate a list of elements in the dataset:
ELEMENTS = set()
for v in data_df['Composition'].values:
    ELEMENTS |= set(eval(v).keys())
ELEMENTS = sorted(ELEMENTS)

def vectorize_composition(composition, elements):
    """ converts an elemental composition dict to a vector. """
    total_n = sum(composition.values())
    vec = np.zeros(len(elements))
    for elem, n in composition.items():
        if elem in elements:
            vec[elements.index(elem)] = n/total_n
    return vec

print('Number of elements:', len(ELEMENTS))
```
Now that we can convert the checmical composition of each material into a feature vector, we will write another function that parses each row in the DataFrame:
```{code-cell}

def parse_data_vector(row):
    """ parses data from a dataframe row """
    
    # parse the composition dict:
    composition_dict = eval(row['Composition'])
    
    # parse feature vector (x):
    x_vector = np.concatenate([
        vectorize_composition(composition_dict, ELEMENTS),
        np.array([ row['Pressure (GPa)'] ]),
        np.array([ row['Tc (K)'] ]),
    ])
     
    return x_vector

```

Finally, we parse each row of the DataFrame with `parse_data_row()` and standardize the data with `StandardScaler`:

```{code-cell}
from sklearn.preprocessing import StandardScaler

data_x = np.array([
    parse_data_vector(row)
    for _, row in data_df.iterrows()
])

print(data_x.shape)

# standardize data:
scaler = StandardScaler()
data_z = scaler.fit_transform(data_x)

# extract a matrix of only compositions:
composition_z = data_z[:,:-2]
```

## Estimate the Empirical Distribution of $T_c$

To understand how $T_c$ values are distributed among the known superconductors in our dataset, we will use kernel density estimation (KDE). This non-parametric method lets us estimate a smooth probability density function with respect to $T_c$. Instead of using the basic KDE module in `scipy.stats`, we will use the more advanced module that is available in [`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html):

```{code-cell}
from sklearn.neighbors import KernelDensity

Tc_data = data_x[:,-1]
Tc_kde = KernelDensity(kernel='gaussian').fit(Tc_data.reshape(-1,1))
```

Let's visualize the distribution, and inspect how the $T_c$ values are distributed. Since 40 K is believed to be the limit of [low-temperature conventional superconductivity](https://en.wikipedia.org/wiki/BCS_theory), let's see how the data is distributed around this point:

```{code-cell}
:tags: [hide-output]
import matplotlib.pyplot as plt

# generate points to evaluate the KDE model:
Tc_eval_pts = np.linspace(0, np.max(Tc_data), 1000)

# determine the standardized density of the distribution:
Tc_density = np.exp(Tc_kde.score_samples(
                        Tc_eval_pts.reshape(-1,1)))
Tc_density /= np.trapz(Tc_density, Tc_eval_pts)

plt.figure()
plt.plot(Tc_eval_pts, Tc_density, label='KDE Distribution')
plt.fill_between(Tc_eval_pts, Tc_density, alpha=0.2)
plt.axvline(
    40,  linestyle=':', color='r',
    label='Limit of Conventional\nSuperconductivity (40 K)')
plt.grid()
plt.xlabel('Temperature (K)')
plt.ylabel('Probability Density')
plt.legend()
plt.xlim(0,np.max(Tc_data))
plt.show()
```

### Identifying Outliers in the Distribution:

Materials with exremely high or low $T_c$ values may represent outliers or materials that exhibit unique physical properties. Let's take a look at any outlying data points by identifying the least probable data points (i.e., the lowest log-likelihood under the KDE model). These can be studied further to see if they reveal special physics or data issues.

To score the log-likelihood of each datapoint, we use the `score_samples()` function and select the $N = 10$ points that have the least likelihood:

```{code-cell}
N_outliers = 10

log_likelihoods = Tc_kde.score_samples(Tc_data.reshape(-1,1))
cutoff = np.sort(log_likelihoods)[N_outliers]

outlier_df = data_df[log_likelihoods < cutoff]
display(outlier_df[['Material','Tc (K)','Pressure (GPa)','DOI']])
```

## Correlations of Composition with $T_c$

To explore which elements are most strongly associated with superconductivity, we will compute the covariance between each compositional feature and the standardized $T_c$. A heatmap on the periodic table then visually shows which elements are associated with increases or decreases in the critical temperature when they are added to a material.

First, we compute the coveriance matrix of the data and extract the column corresponding to the $T_c$ of each material:

```{code-cell}
cov_matrix = np.cov(data_z.T)
Tc_covs = cov_matrix[:-2,-1]

```

Next, we visualize the results using the [`periodic_table_heatmap()`](https://pymatgen.org/pymatgen.util.html#pymatgen.util.plotting.periodic_table_heatmap) function from Pymatgen.
```{code-cell}
:tags: [hide-output]

from pymatgen.util.plotting import periodic_table_heatmap

import matplotlib.pyplot as plt
import matplotlib
# convert Tc covariances to an (element -> cov.) dictionary:
Tc_element_covs = {
    elem : cov
    for elem, cov in zip(ELEMENTS, Tc_covs)
}

# plot covariance periodic table heatmap:
max_cov = np.max(np.abs(Tc_covs))
heatmap = periodic_table_heatmap(
    Tc_element_covs, 
    cmap='coolwarm', 
    symbol_fontsize=20,
    cmap_range=(-max_cov, max_cov),
    blank_color=matplotlib.colormaps['coolwarm'](0.5)
)
plt.title('Correlation of Element Composition with $T_c$', fontsize='20')
plt.show()
```

## Identifying Clusters in the Data

As we have learned in the previous section, dimensionality reduction techniques like PCA allow us to summarize high-dimensional data using a smaller number of principal components. We project the data into this reduced space and then use k-means clustering to group similar superconductors together. These clusters may reflect different material families or classes of superconductors.

```{code-cell}
from sklearn.decomposition import PCA

full_pca = PCA(n_components=data_z.shape[1])
full_pca.fit(data_z)
lambdas = full_pca.explained_variance_
```

```{code-cell}
# cutoff (selected by inspecting the plot below)
pca_cutoff = 10
```
```{code-cell}
:tags: [hide-input]
# plot explained variance vs. principal component:
plt.figure()
plt.bar(np.arange(1,len(lambdas)+1),lambdas)
plt.ylabel(r'Explained Variance ($\lambda$)')
plt.axvline(pca_cutoff, linestyle='--', color='r', 
            label=f'Selected Cutoff: {pca_cutoff}')
plt.xlabel('Principal Component')
plt.show()
```

```{code-cell}
:tags: [hide-output]
pca = PCA(n_components=pca_cutoff)
data_u = pca.fit_transform(data_z)

plt.figure()
plt.grid()
plt.scatter(data_u[:,0], data_u[:,1], marker='+')
plt.xlabel(r'$u_1$')
plt.ylabel(r'$u_2$')
plt.title('Projection onto Principal Components')
plt.show()
```

```{code-cell}
from sklearn.cluster import KMeans

# number of clusters to find:
n_clusters = 6
kmeans = KMeans(n_clusters, n_init='auto')
kmeans.fit(data_u)
assignments = kmeans.predict(data_u)

clusters = {}
for i in range(n_clusters):
    clusters[i] = data_u[assignments == i]
```

```{code-cell}
:tags: [hide-input]
plt.figure()
for i, cluster in clusters.items():
   plt.scatter(cluster[:,0], cluster[:,1], marker='+', 
               label=f'Cluster {i}')
plt.xlabel(r'$u_1$')
plt.ylabel(r'$u_2$')
plt.title('Principal Component Clusters')
plt.legend()
plt.show()
```

To interpret the clusters, we display representative samples from each. You might notice that some clusters are enriched in certain elements or tend to have higher or lower $T_c$. These clusters could guide targeted exploration in materials discovery:

```{code-cell}
:tags: [hide-cell]

# show some samples of each cluster:
for i in clusters.keys():

    # obtain cluster dataframe:
    cluster_df = data_df[assignments == i]
    sample_df = cluster_df[
        ['Material', 'Tc (K)', 'Pressure (GPa)', 'DOI']
    ].sample(5)

    # show sample dataframe:
    display(f'Cluster {i} Examples:')
    display(sample_df)
```
