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

# Application: Bandgap Prediction

## Data Features

## Loading the Dataset

```{code-cell}
:tags: [hide-input]
import pandas as pd

# load dataset into a pandas DataFrame:
BANDGAP_CSV = 'bandgaps.csv'
data_df = pd.read_csv(BANDGAP_CSV)

# show dataframe in notebook:
display(data_df)
```

```{code-cell}
:tags: [hide-input]

# Generate a list of elements in the dataset:
ELEMENTS = set()
for v in data_df['composition'].values:
    ELEMENTS |= set(eval(v).keys())
ELEMENTS = sorted(ELEMENTS)

# Generate a list of the crystal systems in the dataset:
CRYSTAL_SYSTEMS = sorted(set(data_df['crystal_system']))

# print the sizes of ELEMENTS and CRYSTAL_SYSTEMS
print('Number of elements:', len(ELEMENTS))
print('Number of crystal systems:', len(CRYSTAL_SYSTEMS))
```

```{code-cell}
import numpy as np

def vectorize_composition(composition, elements):
    """ converts an elemental composition dict to a vector. """
    total_n = sum(composition.values())
    vec = np.zeros(len(elements))
    for elem, n in composition.items():
        if elem in elements:
            vec[elements.index(elem)] = n/total_n
    return vec

def vectorize_crystal_system(crystal_system, systems):
    """ converts a crystal system to a vector. """
    vec = np.zeros(len(systems))
    if crystal_system in systems:
        vec[systems.index(crystal_system)] = 1.0
        
    return vec
```

```{code-cell}:
:tags: [hide-input]
# generate an example of a composition vector:
test_comp = {'Si': 1, 'O': 2}
print('Example of a composition vector:')
print(vectorize_composition(
            test_comp, 
            elements=['C', 'O', 'Si']))

# generate an example of a crystal system vector:
test_system = 'Hexagonal'
print('Example of a crystal system vector:')
print(vectorize_crystal_system(
            test_system, 
            systems=['Cubic', 'Hexagonal']))
```

```{code-cell}
def parse_data_vector(row):
    """ parses x and y vectors from a dataframe row """
    
    # parse whether or not the bandgap is direct: 
    bandgap_direct = 1.0 if row['bandgap_direct'] == 'True' else -1.0
    
    # parse the composition dict:
    composition_dict = eval(row['composition'])
    
    # parse feature vector (x):
    x_vector = np.concatenate([
        vectorize_composition(composition_dict, ELEMENTS),
        vectorize_crystal_system(row['crystal_system'], CRYSTAL_SYSTEMS),
        np.array([ row['volume'] ]),
        np.array([ row['density'] ]),
        np.array([ row['formation_energy_per_atom'] ]),
    ])
    
    # parse label vector (y):
    y_vector = np.concatenate([
        np.array([ row['band_gap'] ]),
        np.array([ bandgap_direct ])
    ])
    
    return x_vector, y_vector
```

```{code-cell}
:tags: [hide-input]

# parse x and y vectors from dataframe rows:
data_x, data_y = [], []
for i, row in data_df.iterrows():
    x_vector, y_vector = parse_data_vector(row)
    data_x.append(x_vector)
    data_y.append(y_vector)

# convert x and y to numpy arrays:
data_x = np.array(data_x)
data_y = np.array(data_y)

print('data_x shape:', data_x.shape)
print('data_y shape:', data_y.shape)
```

```{code-cell}
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_val_test_split(data_x, data_y, split=(0.8,0.1,0.1)):
    """ splits data into train, validation, and test sets. """
    
    # split train and nontrain data:
    train_x, nontrain_x, train_y, nontrain_y = \
        train_test_split(
            data_x, data_y, train_size=split[0]/sum(split))
    
    # split validation and test data:
    val_x, test_x, val_y, test_y = \
        train_test_split(
            data_x, data_y, 
            test_size=split[2]/(split[1]+split[2]))
    
    return (train_x, val_x, test_x), \
           (train_y, val_y, test_y)
    
def normalize(train_x, val_x, test_x):
    """ normalizes a dataset. """
    
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)
    return scaler, train_z, val_z, test_z

```
## Classifying Metals and Non-Metals

```{code-cell}
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

metals_y = np.array([ 1.0 if y[0] <= 0 else -1 for y in data_y])

metal_subsets_x, metal_subsets_y = \
    train_val_test_split(data_x, metals_y)

train_x, val_x, test_x = metal_subsets_x
train_y, val_y, test_y = metal_subsets_y

scaler, train_z, val_z, test_z = \
    normalize(train_x, val_x, test_x)

model = RidgeClassifier(alpha=20)
model.fit(train_z, train_y)

train_yhat = model.predict(train_z)
val_yhat = model.predict(val_z)

# compute accuracy:
cm = confusion_matrix(val_y, val_yhat)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# display confusion matrix:
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()
plt.gca().set_yticklabels(['Non Metal', 'Metal'])
plt.gca().set_xticklabels(['Non Metal', 'Metal'])
plt.show()
```

## Estimating the Bandgap of Non-Metals:

```{code-cell}
:tags: [hide-input]
bandgap_x = data_x[data_y[:,0] > 0]
bandgap_y = data_y[(data_y[:,0] > 0),0]

#direct_gap_y = data_y[(data_y[:,0] > 0),1]

print('bandgap_x shape:', bandgap_x.shape)
print('bandgap_y shape:', bandgap_y.shape)
```

### Ridge Regression Model:
```{code-cell}
:tags: [hide-input]

from sklearn.linear_model import Ridge

bandgap_subsets_x, bandgap_subsets_y = \
    train_val_test_split(bandgap_x, bandgap_y)

train_x, val_x, test_x = bandgap_subsets_x
train_y, val_y, test_y = bandgap_subsets_y

scaler, train_z, val_z, test_z = \
    normalize(train_x, val_x, test_x)

ridge_model = Ridge(alpha=0.01)
ridge_model.fit(train_z, train_y)

train_yhat = ridge_model.predict(train_z)
val_yhat = ridge_model.predict(val_z)

train_mse = np.mean((train_yhat - train_y)**2)
val_mse = np.mean((val_yhat - val_y)**2)

print('training stddev:', np.std(train_y))
print('training MSE:', train_mse)
print('validation MSE:', val_mse)
print('validation MSE/stddev:', val_mse/np.std(train_y))
```

### Nearest Neighbor Regression Model

```{code-cell}
:tags: [hide-input]
from sklearn.neighbors import KNeighborsRegressor

nn_model = KNeighborsRegressor(n_neighbors=5,
                               weights='distance',
                               algorithm='ball_tree')
nn_model.fit(train_z, train_y)

train_yhat = nn_model.predict(train_z)
val_yhat = nn_model.predict(val_z)

train_mse = np.mean((train_yhat - train_y)**2)
val_mse = np.mean((val_yhat - val_y)**2)

print('training stddev:', np.std(train_y))
print('training MSE:', train_mse)
print('validation MSE:', val_mse)
print('validation MSE/stddev:', val_mse/np.std(train_y))
```

## Exercises

:::{dropdown} Exercise 1: Classifying Direct vs. Indirect Bandgaps

```
direct_gap_x = data_x[data_y[:,0] > 0]
direct_gap_y = data_y[(data_y[:,0] > 0),1]

print('direct_gap_x shape:', direct_gap_x.shape)
print('direct_gap_y shape:', direct_gap_y.shape)
```
:::

### Solutions:

#### Exercise 1: Classifying Direct vs. Indirect Bandgaps

```{code-cell}
:tags: [hide-cell]

import numpy as np
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt

direct_gap_x = data_x[data_y[:,0] > 0]
direct_gap_y = data_y[(data_y[:,0] > 0),1]

print('direct_gap_x shape:', direct_gap_x.shape)
print('direct_gap_y shape:', direct_gap_y.shape)
```

