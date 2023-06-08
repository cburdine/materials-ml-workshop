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

# Classes and Object-Oriented Programming

Objects and classes are fundamental concepts in [_Object-Oriented Programming_](https://en.wikipedia.org/wiki/Object-oriented_programming), which is a paradigm of programming that involves the encapsulation of data and operations performable on data into entities called _objects_. Objects provide a way to model real-world entities, concepts, or abstract ideas in a program. They enable the development of more organized, maintainable, and extensible code by grouping related data and behavior into self-contained units. In this section we will talk about how we can design and use obects in our Python code by writing _classes_. Classes serve as blueprints for objects- they define the data and behaviors that an object has when it is created during the execution of a program. In this section, we will give a brief introduction to objects, classes, and how they are useful in Python.

# Objects in Python

Although you ma not have known it, you have encountered objects before in Python- in the form of Python's built-int types such as `str`, `list`, `dict`, etc. We have also encountered some functions associated with these objects, which we have accessed using the syntax `object.function()`. For example:

```{code-cell}
print('ThisIsAString'.to_lower())

print([1,2,3].pop())

print({ 'aKey' : 'aValue' }.keys())

```

# Python Classes

# Exercises
