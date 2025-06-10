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

# Data Analysis and Visualization

Now that we have learned about the Numpy and Scipy packages, let's take a look at some packages that can help us with analyzing and visualizing data. Although there are several Python packages that can assist with these tasks, we will focus on the two most popular packages: _Pandas_ and _Matplotlib_. Pandas (`pandas`) is a package that provides an interface for working with large heterogeneous datasets using array-like structures called DataFrames. Matplotlib (`matplotlib`) is the most popular data visualization tool in Python, with an interface inspired by the plotting functionality in MATLAB. In this section, we will discuss how to analyze data with the `pandas` package, and discuss how to plot and visualize data with `matplotlib` in the next section.

## The Pandas Package

Pandas is an open-source Python package for data manipulation and analysis. The name of package is (sadly) not named after the fuzzy black and white mammal. It is a contraction of "panel datasets", which is a term from econometrics that refers to the type of series-based data commonly used in that field. Much like how the `numpy` package is centered around the `numpy.ndarray` data type, Pandas is centered around an array-like data type called `pandas.DataFrame`.

When using the Pandas package, it is customary to import it with the alias `pd`:

```{code-cell}
import pandas as pd
```

## Working with Pandas DataFrames

A DataFrame is a two-dimensional table-like structure with labeled rows and columns. It is similar to a spreadsheet or [SQL table](https://en.wikipedia.org/wiki/SQL). The easiest way to create a DataFrame is by constructing one from a Python dictionary as follows:

```{code-cell}
# Data on the first four elements of the periodic table:
elements_data = {
    'Element' : ['H', 'He', 'Li', 'Be'],
    'Atomic Number' : [ 1, 2, 3, 4 ],
    'Mass' : [ 1.008, 4.002, 6.940, 9.012],
    'Electronegativity' : [ 2.20, 0.0, 0.98, 1.57 ]
}

# construct dataframe from data dictionary:
df = pd.DataFrame(elements_data)
```

In a Jupyter Notebook, we can display a Pandas DataFrame using the `display` function:

```{code-cell}
display(df)
```

## Data Manipulation:

Pandas provides various functions and methods that make the manipulation and transformation of data relatively simple. Using square brackets (i.e. `[`...`]`) and the methods in the Dataframe class, we can index the DataFrame by row or column or even ranges of rows and columns. For example:

```{code-cell}
:tags: [hide-output]
# access a single column:
display(df['Element'])

# access multiple columns:
display(df[ ['Element', 'Mass'] ])

# access a single row:
display(df.iloc[1])

# access a single value:
display(df.at[1,'Mass'])

# access a range of rows:
display(df.iloc[1:3])

# access multiple rows and columns simultaneously:
display(df.iloc[1:3, :2])
```

From inspecting the output above, we notice that when a range of rows or columns is accessed, the returned result is a `pandas.DataFrame`. However, whenever we access a row or a column of a Dataframe, the result is a 1D sequence of values called a _Series_, not a DataFrame. To access the values of a series, we simply use square brackets

```{code-cell}
# extract series (i.e. row or column) from dataframe:
element_series = df['Element']
helium_series = df.iloc[1]

# verify the returned types are a Series object:
print(type(element_series))
print(type(helium_series))

# access values in series:
print(element_series[0])
print(helium_series['Mass'])
```

Sometimes, we might want to extract Series data as a Numpy array. This can be done by simply constructing a Numpy array from the Series object:

```{code-cell}
import numpy as np

# get the 'Mass' column and convert it to a numpy array:
mass_series = df['Mass']
mass_array = np.array(mass_series)

print(mass_array)
```

We can also filter the rows of DataFrames using Boolean indexing. This feature is similar to how Numpy arrays can be filtered:

```{code-cell}
# only show electronegative elements:
filtered_df = df[ df['Electronegativity'] > 0 ]

display(filtered_df)
```

We can also add columns to the DataFrame by assigning a list (or Numpy array) of values to the new column name. For example:

```{code-cell}
# add group and period columns to the dataframe:
df['Group'] = [ 1, 18, 1, 2 ]
df['Period'] = np.array([ 1, 1, 2, 2 ])

display(df)
```

## Transforming Data

A crucial part of working with Pandas Dataframes is the transformation of data. Usually this involves applying some mathematical function to a Dataframe column and storing the result in a new Dataframe column. To show how this is done in Pandas, let's write a function called `approximate_mass`, which (naively) approximates atomic mass as the atomic number times the sum of the proton and neutron masses (in atomic mass units). We can apply the function using the [`apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html?highlight=dataframe%20apply#pandas.DataFrame.apply) function on either a Series or DataFrame object:

```{code-cell}
from scipy.constants import proton_mass, neutron_mass, m_u

# define an approximation function:
def approximate_mass(atomic_number):
    """ Naively approximates atomic mass """
    return atomic_number * (proton_mass + neutron_mass)/ m_u

# add an "Estimated Mass" column to the dataframe:
df['Estimated Mass'] = df['Atomic Number'].apply(approximate_mass)

display(df)
```

## Analyzing Data

Pandas provides a wide range of functions for analyzing data. The quickest way to obtain summary statistics of numerical columns in a Dataframe is by using the `describe` method:

```{code-cell}
df.describe()
```
This function reports key statistics, such as total counts, mean, standard deviation (std), min, max, etc. We can obtain these values manually using the method with the respective name (e.g. `df.mean()`). Below, we give some examples of how to compute these statistics individually:

```{code-cell}
:tags: [hide-output]
# compute mean of the 'Mass' column:
display(df['Mass'].mean())

# compute standard deviations of mass and electronegativity:
display(df[['Mass', 'Electronegativity']].std())

# compute the mean of electronegativity grouped by 'Group':
display(df.groupby('Group')['Electronegativity'].mean())

```

## Importing and Exporting Data

Pandas supports reading and writing data from/to various file formats, including CSV (_comma-separated values), Excel spreadsheets, SQL databases, and more. You can use functions `like read_csv`, `to_csv`, `read_excel`, `to_excel`, `read_sql`, and `to_sql` to handle data input and output operations. In this workshop, we will use data that is primarily written in the CSV format. For example:

```{code-cell}
# export dataframe to CSV file:
df.to_csv('elements_data.csv', index=False)

# import the exported csv back into a Dataframe:
imported_df = pd.read_csv('elements_data.csv')

# display imported DataFrame:
display(imported_df)
```

## Exercises

:::{dropdown} Exercise 1: Exploring the Periodic Table
For this exercise, we will be working with a large dataset with data about elements of the Periodic Table. First, download the dataset [here](https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee/archive/1d92663004489a5b6926e944c1b3d9ec5c40900e.zip) and extract the `Periodic Table of Elements.csv` file into the same folder as your Jupyter Notebook. Then, you should be able to import the data into your Jupyter Notebook as follows:

```
import pandas as pd

# Load periodic table dataframe:
ptable_df = pd.read_csv('Periodic Table of Elements.csv')

# Display dataframe columns:
display(ptable_df.columns) 

# Display dataframe:
display(ptable_df)
```

Using this data, answer the following questions:

1. What fraction of elements of the Periodic Table were discovered before 1900?
2. Which elements have at least 100 isotopes?
3. What is the average atomic mass of the radioactive elements?

---
Data used in this exercise was obtained from [GoodmanSciences](https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee).
:::

### Solutions

#### Exercise 1: Exploring the Periodic Table

```{code-cell}
:tags: [hide-cell]
import numpy as np
import pandas as pd

# Load periodic table dataframe:
ptable_df = pd.read_csv('Periodic Table of Elements.csv')

# Display dataframe columns:
display(ptable_df.columns) 

# Display dataframe:
display(ptable_df)


# Question 1:
print('What fraction of elements of the periodic table was discovered before 1900?')

year_discovered = ptable_df['Year']
frac = year_discovered[year_discovered < 1900].count() / year_discovered.count()
print(frac)


# Question 2:
print('\nWhich elements have at least 100 isotopes?')

n_isotopes = ptable_df['NumberOfIsotopes']
max_isotope_df = ptable_df[n_isotopes >= 100]
print(max_isotope_df[['Element','NumberOfIsotopes']])


# Question 3:
print('\nWhat is the average atomic mass of radioactive elements?')
radioactive_elements = ptable_df[ptable_df['Radioactive'] == 'yes']
print(radioactive_elements['AtomicMass'].mean())
```
