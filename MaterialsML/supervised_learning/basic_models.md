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

```{code-cell}
:tags: [hide-input]
import pandas as pd

# make pandas dataframes interactive (optional):
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)

# load dataset into a pandas DataFrame:
PEROVSKITE_CSV = './perovskites.csv'
perovskite_df = pd.read_csv(PEROVSKITE_CSV)

# show dataframe in notebook:
display(perovskite_df)
```

## Choosing Meaningful Features



## Nearest-Neighbor Models

### Classification



### Regression

### Kernel Density Estimation


## Exercises

### Solutions

## Data Sources

The Data used in this section was obtained from [Mendeley Data](https://data.mendeley.com/datasets/dfsf6n6g7y/1) (also hosted on [Kaggle](https://www.kaggle.com/datasets/sayansh001/crystal-structure-classification?resource=download)). You can cite this dataset as follows:
```
Thomas, Tiju; Behara, Santosh; Poonawala, Taher (2020), “Data for: Crystal structure classification in ABO3 perovskites via machine learning”, Mendeley Data, V1, doi: 10.17632/dfsf6n6g7y.1
```
