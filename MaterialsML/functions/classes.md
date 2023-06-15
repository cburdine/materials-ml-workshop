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
        """ Constructs a Dog instance with given name """
        self.name = dog_name

```

First, observe that the declaration of a class begins with the keyword `class` followed by the class name and a colon (`:`). Underneath it, we have added a docstring (enclosed in `"""`...`"""`) describing what the class does. After this, we have an indented class body, which includes the class constructor function, which is always named  `__init__`. When writing the constructor, the first parameter is always the keyword `self` (which represents the new object being constructed). The constructor parameters after `self` are any additional parameters (e.g. `dog_name` in the example above) we might need for constructing the object. In the `Dog` class constructor, the parameter `dog_name` is passed as an additional parameter and its value is assigned to the variable `self.name` inside the `__init__` function. Inside the constructor, the line `self.name = dog_name` creates an instance variable called `name` associated with the new Dog object being constructed.

:::{tip}
In Python it is considered good practice to capitalize the name of classes and avoid using underscores (`_`) in class names in order to distinguish them from other built-in types. For example, `Dog`, `MyClass` and `MyClassWithALongName` are all class names that abide by this convention.
:::

We can construct a new object (specifically, an instance of the `Dog` class) by calling the class name as if it were the class `__init__` function, but without the `self` argument:

```{code-cell}
# construct a Dog object with name 'Snoopy':
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

We can also add methods (i.e. functions associated with a class) by adding more functions to the class body. Like `__init__`, these functions should have `self` as the first argument. To illustrate this, let's expand our `Dog` class by adding some instance variables and methods that use them:

```{code-cell}
class Dog:
    """ This class represents a pet dog """

    # Constructor:
    
    def __init__(self, dog_name, dog_age=1):
        """ Constructs a Dog instance with given name """
        self.name = dog_name
        self.age = dog_age
        self.tricks = []

    # Additional Methods:
    
    def human_years_age(self):
        """ returns age in human-scale years (7x age) """
        return 7 * self.age

    def add_trick(self, trick):
        """ Adds a trick to this dog's routine of tricks """
        self.tricks.append(trick)

    def do_tricks(self):
        """ Performs this dog's routine of tricks """
        for trick in self.tricks:
            print(self.name, ':', trick)

```

```{code-cell}
# create a dog named Buddy:
my_dog = Dog('Buddy', 2)

# print dog name and age:
print('Name:', my_dog.name)
print('Age:', my_dog.age)

# print age in human years:
print('Age in human-scale years:', my_dog.human_years_age())

# add some tricks:
my_dog.add_trick('Sit')
my_dog.add_trick('Shake')
my_dog.add_trick('Roll Over')

# perform all added tricks:
my_dog.do_tricks()

```

:::{admonition} Advanced Tip: Special Class Methods
:class: tip, dropdown

There are some special methods in Python that can define class behaviors for built-in Python operations (such as the `==` operator), and functions (such as `print`). 

```
def __repr__(self):
    """ 
        Returns a string that is printed when print() is
        called on an object of this class
    """
    return '(string form of object)'

def __eq__(self, other):
    """
        Defines behavior when an instance of this class is
        compared to another object with the `==` operator.
    """
    return (self.variable == other.variable)
```

You can read more about other special features of classes in the [Official Python Documentation](https://docs.python.org/3/tutorial/classes.html?highlight=special%20methods).
:::

## Exercises

:::{dropdown} Exercise 1: Inorganic Material Class
Design and write a Python class called `InorganicMaterial` that represents a crystalline inorganic material. This class should have the following instance variables:

* `atoms`: a dictionary with element-number pairs representing the atomic composition of the material unit cell. For example: `{'Si':1, 'O':2}` represents SiO$_2$. (_default value_: `{}`)
* `crystal_system`: a string representing the [crystal system](https://en.wikipedia.org/wiki/Crystal_system) of the material. (_default value_: `'cubic'`)
* `lattice_constants`: a tuple of numbers representing the lattice constants of the crystal in Å. (_default value_: `()`)

Your class should also have the following methods:

* A constructor with all instance variables as parameters (each parameter should have the default values from above)

* `add_atoms(self,element,n=1)`: 
Adds $n$ atoms of element `element` to the unit cell composition (represented by `atoms`).

* `remove_atoms(self,element,n=1)`: 
Removes $n$ atoms of element `element` to the unit cell composition. (If `element` is not contained in `atoms`, this does nothing).

* `total_atoms(self)`:
Returns the total number of atoms in the unit cell.

* `get_formula(self)`: 
Returns a string representation of a material's formula (i.e. `'SiO2'`) where the elements in the formula are ordered alphabetically with the exception of oxygen, which is placed at the end of the formula if present. You do not need to reduce the ratio of elements in this formula, but if an element occurs with count $n=1$ in the unit cell, the $1$ should not be included (i.e. `'SiO2'`, not `'Si1O2'`).

To test your new class, create an instance of your favorite inorganic oxide material.

---

_Hint_: To sort a list of strings alphabetically, use the `sorted` function:
```
sorted([ 'H', 'He', 'Li', 'Be' ]) #  --> ['Be', 'H', 'He', 'Li']
```

Also, recall that you can convert numbers to their respective string representation using the `str()` function:

```
str(12345) # --> '12345'
```
:::

### Solutions:

#### Exercise 1: Inorganic Material Class

```{code-cell}
:tags: [hide-cell]

class InorganicMaterial:
    """ 
    This class represents an Inorganic Material with a crystalline structure.

    Instance variables:
        atoms: a dictionary of element-count pairs, representing the unit cell
        crystal_system: a string representing the materials crystal system
        lattice_constants: a tuple of floats representing the unit cell lattice constants
    """

    def __init__(self, atoms={}, crystal_system='cubic', lattice_constants=()):
        """ Constructor for an Inorganic material """
                
        self.atoms = atoms
        self.crystal_system = crystal_system
        self.lattice_constants = lattice_constants

    def add_atoms(self, element, n=1):
        """ adds an element with count n to the unit cell """

        # increase the count of the element (add it, if it is not in atoms):
        if element in self.atoms:
            self.atoms[element] += n
        else:
            self.atoms[element] = n

    def remove_atoms(self, element, n=1):
        """ removes n counts of an element in the unit cell """

        # decrement the count of the element:
        if element in self.atoms:
            self.atoms[element] -= n
            
            # remove element if it causes the count to be non-positive:
            if self.atoms[element] <= 0:
                self.atoms.pop(element)

    def total_atoms(self):
        """ returns the total number of atoms in the unit cell """
        return sum(self.atoms.values())

    def get_formula(self):
        """ returns the unit cell formula in string form """

        # determine the order of elements in the formula:
        ordered_elements = sorted([ 
            elem for elem in self.atoms.keys() if elem != 'O' 
        ])
        if 'O' in self.atoms:
            ordered_elements.append('O')
    
        # construct formula
        formula = ''
        for element in ordered_elements:
            
            # only include elements with n > 0 in the formula:
            n = self.atoms[element]
            if n > 0:
                formula += (element + (str(n) if n > 1 else ''))
        
        return formula

# construct an inorganic material instance:
material = InorganicMaterial(atoms={'Y':1, 'Ba':2, 'Cu':3, 'O':7 },
                             crystal_system='tetragonal',
                             lattice_constants=(3.89, 3.89, 12.12))

# print formula and total atoms:
print(material.get_formula())
print('Total atoms:', material.total_atoms())

# add and remove some atoms:
material.add_atoms('Hg')
material.add_atoms('Ca',2)
material.add_atoms('O')
material.remove_atoms('Y')

# print new formula and total atoms:
print(material.get_formula())
print('Total atoms:', material.total_atoms())
```
