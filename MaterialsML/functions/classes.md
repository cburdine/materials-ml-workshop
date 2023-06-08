---
jupytext:
  formats: ipynb,md:myst
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

# Classes and Object-Oriented Programming

Objects and classes are fundamental concepts in [_Object-Oriented Programming_](https://en.wikipedia.org/wiki/Object-oriented_programming), which is a paradigm of programming that involves the encapsulation of data and operations performable on data into entities called _objects_. Objects provide a way to model real-world entities, concepts, or abstract ideas in a program. They enable the development of more organized, maintainable, and extensible code by grouping related data and behavior into self-contained units. In this section we will talk about how we can design and use obects in our Python code by writing _classes_. Classes serve as blueprints for objects- they define the data and behaviors that an object has when it is created during the execution of a program. In this section, we will give a brief introduction to objects, classes, and how they are useful in Python.

# Objects in Python

Although you may not have known it, you have encountered Python objects before in the form of Python's built-in types, such as `str`, `list`, `dict`, etc. We have also encountered some functions associated with these objects, which we have accessed using the syntax `object.function()`. For example:

```{code-cell}
str_example = 'ThisIsAString'
list_example = [1,2,3]
dict_example = { 'aKey' : 'aValue' }

# the `lower` function is associated with a `str` object:
print(str_example.lower())

# the `append` function is associated with a `list` object:
list_example.append(4)
print(list_example)

# The `keys` function is associated with a `dict` object:
print(list(dict_example.keys()))
```

Note that the values returned or modified in each of the function calls are specific to each object instance (i.e `str_example`, `list_example`, `dict_example`). These functions that are closely associated with an object in are called _methods_. We will also see that objects can have variables associated with them, called _instance variables_. These variables can also be accessed and assigned values just like regular variables in Python. 

# Python Classes

The easiest way to understand how objects, methods, and instance variables work it to learn how to create your own custom Python objects. To create our own objects, we must first define a kind-of "blueprint" for an object called a _class_. A class is a block of code that describes the functions (i.e. methods) and variables (i.e. instance variables) associated with each object we create in our code. The most important method that we define in a class is a _constructor_, which in Python is a method with the name `__init__` used to make a new object of the associated class. To illustrate this, let's create a simple class called `Dog`:

```{code-cell}
class Dog:
    """ This class represents a pet dog """
       
    def __init__(self, dog_name):
        self.name = dog_name

```

First, observe that the declaration of a class begins with the keyword `class` followed by the class name and a colon (`:`). Underneath it, we have added a docstring (enclosed in `"""`...`"""`) describing what the class does. After this, we have an indented class body, which includes the class constructor function, which is always named  `__init__`. When writing the constructor, the first parameter is always the keyword `self` (which represents the new object being constructed). The constructor parameters after `self` are any additional parameters (e.g. `dog_name` in the example above) we might need for constructing the object. In the `Dog` class constructor, the parameter `dog_name` is passed as an additional parameter and its value is assigned to the variable `self.name` inside the `__init__` function. Inside the constructor, the line `self.name = dog_name` creates an instance variable called `name` associated with the new Dog object being constructed.

:::{tip}
In Python it is considered good practice to capitalize the name of classes and avoid using underscores (`_`) in class names in order to distinguish them from other built-in types. For example, `Dog`, `MyClass` and `MyClassWithALongName` are all class names that abide by this convention.
:::

We can construct a new object (specifically, an instance of the `Dog` class) by calling the class name as if it were the class `__init__` function, but without the `self` argument:

```{code-cell}
# construct a Dog object with name 'Snoopy'
my_dog = Dog('Snoopy') # <-- this calls the __init__ constructor

# print the type of my_dog:
print(type(my_dog))
```

# Instance Variables

In the `Dog` class construcor defined above, we created the instance variable `name`, which we assigned the value of the constructor parameter `dog_name`. This instance variable is specific to each instance of the `Dog` class we construct. To illustrate this, let's create some `Dog` instances with different `name` values, and show that modifying the instance variables of one instance does not affect the other:

```{code-cell}
# construct two different Dog objects:
my_dog = Dog('Snoopy')
my_other_dog = Dog('Fido')

# print the names of the dogs:
print(my_dog.name, my_other_dog.name)

# assign the name of one dog to:
my_other_dog.name = 'Rover'

print(my_dog.name, my_other_dog.name)
```

# Methods

# Exercises
