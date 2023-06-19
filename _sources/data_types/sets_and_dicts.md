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

# Sets and Dictionaries

Now that we have talked about lists and tuples and Python, let's talk about two more of the built-in Python data types: _sets_ and _dictionaries_. Unlike lists and tuples, which deal with sequentially ordered data, sets and dictionaries deal with unordered data. These data types are especially useful for finding unique elements or assigning values to a collection of unique keys. In this section, we will take a look at each of these types and learn how they are used in Python programming.

## Python Sets

The `set` data type stores an unordered collection of unique elements (just like the [mathematical definition of a set](https://en.wikipedia.org/wiki/Set_(mathematics))). Unlike lists or tuples, sets do not maintain any specific order for their elements. The primary characteristics of sets are their uniqueness and their ability to perform various mathematical set operations efficiently. In Python we create sets by enclosing a sequence of comma-separated values by curly braces (`{`...`}`) or by using the `set()` function. For example:

```{code-cell}
# create a set (duplicate elements will be removed):
example_set = {'A', 'B', 'C', 'A', 'F', 'B'}

# print the set:
print(example_set)
```

Just like mathematical sets, the Python `set` type supports canonical set operations, such as union, intersection, difference, etc. These operations allow you to combine, compare, or manipulate sets efficiently. In Python we can perform the `union`, `intersection`, and `difference` operations using either the function with the same name or the operator shorthand (union: `|`, intersection: `&`, difference: `-`). Below, we give examples of how these operations can be applied:


```{code-cell}
# Create sets of physicists and chemists:
chemists = { 'Bohr', 'Curie', 'Pauling', 'Berzelius' }
physicists = { 'Planck', 'Bohr', 'Pauli', 'Curie', 'Fermi' }

# print out the result of a set union:
# (i.e. people who are chemists or physicists):
print(chemists.union(physicists))
print(chemists | physicists)

# print the result of a set intersection:
# (i.e. people who are chemists and physicists):
print(chemists.intersection(physicists))
print(chemists & physicists)

# print the result of a set difference:
# (i.e. people who are chemists but not physicists):
print(chemists.difference(physicists))
print(chemists - physicists)
```

Similar to lists, sets are mutable data types, meaning you can add or remove elements from them. This is done using the functions `add` and `remove`. However, take note that sets cannot contain mutable data types such as lists. Here are some examples of the `add` and `remove` operations:

```{code-cell}
# build a set of names:
physicists = { 'Planck', 'Bohr', 'Pauli', 'Curie', 'Fermi' }

# add an element to the set:
physicists.add('Einstein')
print(physicists)

# adding an existing element does not change the set:
physicists.add('Bohr')
print(physicists)

# remove an element from the list: 
physicists.remove('Planck')
print(physicists)
```

We can also build and filter sets using list comprehension syntax:

```{code-cell}
# build a set of names:
physicists = { 'Planck', 'Bohr', 'Pauli', 'Curie', 'Fermi' }

# filter set using list comprehension syntax:
subset = { name for name in physicists if len(name) == 5 }
print(subset)
```

## Python Dictionaries

Python dictionaries (`dict`s) are a powerful built-in data type that allows you to store and retrieve data in key-value pairs. Dictionaries are unordered and mutable, making them suitable for a wide range of tasks. In Python, we create dictionaries by enclosing comma separated `key : value` pairs in curly braces (`{`...`}`). Although both the `set` and `dict` data types use curly braces, the key distinguishing factor between them is the use of the colon `:` between key-value pairs in a `dict`. For example:

```{code-cell}
# create a dictionary representing a person:
person = {
    'name' : 'John von Neumann',
    'age' : 53,
    'occupation' : 'Mathematician',
    'birthday' : ('December', 28, 1903)
}

# print the dictionary:
print(person)
```

In the example above, a dictionary named `person` with four key-value pairs is created, representing the name, age, occupation, and birthday associated with the person.

:::{note}
In Python, `{}` denotes an empty dictionary, not an empty set. To construct an empty set, use `set()`:
```
empty_dict = {}
empty_set = set()
```
:::

You can access, add, and modify the values in a dictionary by referencing their corresponding keys. Use square brackets (`[]`)  with the key inside to retrieve the associated value. This syntax is similar to accessing the elements of a list. 

For example, we can access the values of the `person` dict as follows:

```{code-cell}
# print the value of the 'name' key:
print(person['name'])

# print the value of the 'birthday' key:
print(person['birthday'])

# update the value of the 'occupation key':
person['occupation'] = 'Physicist'
print(person['occupation'])

# add the new key 'city' with value 'New Jersey':
person['city'] = 'New Jersey'
print(person['city'])
```

## Dictionary Operations

There are are several useful functions and operations that can be applied to dictionaries. For example:

```{code-cell}
# create an example dictionary:
example_dict = {
    (1,1) : [1],
    (1,2) : [2,2],
    (1,3) : [3,3,3]
}

# use the 'in' operator to check if dict contains a key:
print((2,1) in example_dict)
print((1,2) in example_dict)

# print the dictionary keys:
print(list(example_dict.keys()))

# print the dictionary values:
print(list(example_dict.values()))

# use items() to enumerate over key-value pairs:
for key, value in example_dict.items():
    print(key, '=', value)
```

## Exercises

:::{dropdown} Exercise 1: Scheduling
Suppose you and your colleagues are trying to agree on a time to schedule a meeting. You and each of your colleagues knows the hours of the day that each of you are available to meet, which you have each put into a Python `set` data type. These sets have been organized into a list as follows:

```
# times each person is available to meet:
available_times = [
    {1,2,3,4,6,7}, # person 1's availability
    {2,4,6},       # person 2's availability
    {3,4,5,6,7,8}, # ... etc.
    {1,4,6,7,8},
]
```

Write a Python program that prints out the hours that are present in everyone's availability so that you can agree on a meeting time.

---
_Hint_: Which set operation (`union`, `intersection`, or `difference`) should be applied to each availability?
:::


:::{dropdown} Exercise 2: Most Frequent Number

Suppose you are given a large list of numbers, e.g:
```
numbers = [ 23, 1, 12, 24, 1, 4, -3, 12, 1, 23, 12 ]
```

Write a Python program to find print out the most frequent number in the list and the number of times it appears. (If multiple numbers have the highest frequency, print all of them.)

---
_Hint_: Use a dictionary called `frequencies` to keep track of the frequencies of each number as you iterate through `numbers`. Use the number as the key and the frequency as the value. To get the maximum frequency from the dictionary, use the `max` function:

```
maximum_frequency = max(frequencies.values())
```
:::

### Solutions

#### Exercise 1: Scheduling

```{code-cell}
:tags: [hide-cell]
# times each person is available to meet:
available_times = [
    {1,2,3,4,6,7}, # person 1's availability
    {2,4,6},       # person 2's availability
    {3,4,5,6,7,8}, # ... etc.
    {1,4,6,7,8},
]

# set meeting_times to be the intersection of all schedules:
meeting_times = available_times[0]
for timeset in available_times[1:]:
    meeting_times = meeting_times & timeset

# print the set of possible meeting times:
print('Avaliable meeting times:', meeting_times)
```

#### Exercise 2: Most Frequent Number

```{code-cell}
:tags: [hide-cell]

# Given list of numbers:
numbers = [ 23, 1, 12, 24, 1, 4, -3, 12, 1, 23, 12 ]

# populate a frequency dictionary:
frequencies = {}
for n in numbers:

    # increment the frequency of n:
    if n in frequencies:
        frequencies[n] += 1
    else:
        frequencies[n] = 1

# determine most frequent numbers:
max_frequency = max(frequencies.values())

# extract the most frequent numbers with list comprehension:
most_frequent_numbers = [
    n for n, count in frequencies.items() 
    if count == max_frequency
]

# print most frequent numbers:
print('Most frequent numbers:', most_frequent_numbers)
print('Frequency:', max_frequency)
```
