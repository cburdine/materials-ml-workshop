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

# Python Functions and Classes

_Functions_ are an essential concept in Python and programming in general. They allow you to encapsulate reusable pieces of code into named blocks, which can be called and executed whenever needed. Functions promote code modularity, readability, and reusability, making them a fundamental building block for structuring and organizing programs. Likewise, _Classes_ in Python provide a way of encapsulating both data and the operations performable on it. In this section we will focus on writing Python functions, and see how they can help us write more concise code.

## Function Basics

In mathematics, functions are transformations that map inputs to outputs. The same is true for Python, except for the fact that Python functions can also change the values of variables, modify mutable data types (like lists), etc. So far, we have encountered several of Python's built-in functions, for example:

```{code-cell} ipython3
# create a list:
my_list = [10, 20, 30, 40]

# `len` is a function that takes a list as input and outputs the length:
list_length = len(my_list)
print(list_length)

# `pop` is a function associated with list objects that modifies the list
#  and outputs the last value:
last_value = my_list.pop()

print(last_value, my_list)
```

To create our own function in Python, we use the `def` keyword followed by the function name, a set of parentheses `()` and a colon `:`. The function body is indented below the declaration of the function, similar to `if` statements and `for` loops:

```{code-cell} ipython3
# create a function to print out a greeting:
def greet():
    print('Hello, world!')
```

In the example, we have defined a function `greet` that does not take any parameters (similar to the `list.pop`) function above. To execute a function and perform its associated actions, you call the function by using its name followed by parentheses `()`. If the function has parameters, you can pass values within the parentheses. Since our `greet` function has no parameters, we can call the function (i.e. execute is corresponding block of code) by using its name followed by parentheses `()`:

```{code-cell} ipython3
greet()
```

Upon calling the `greet` function, we see that the `print('Hello, world!')` line is executed, resulting in the output `Hello, world!`.

## Function Parameters: 

Like mathematical functions, Python functions can accept _parameters_, which are variables used to pass values into the function. Parameters are specified within the parentheses of the function definition. To call a function with parameters, we simply put the value of the parameter between the parentheses when the function is called. 

For example, let's re-write our `greet` function to accept a parameter called `name`, and then call the function with the value `'Albert'` as `name`:

```{code-cell} ipython3
# updated function that prints a greeting for a specific name:
def greet(name):
    print('Hello, ' + name + '!')

# call greet with 'Albert':
greet('Albert')
```

Functions can also have multiple parameters, which we can denote using a comma separated list. When calling a function with multiple parameters, be sure that the order of the parameters match the order of the corresponding values:

```{code-cell} ipython3
# This function prints out a greeting of a name (with a title):
def greet(name, title):
    print('Hello, ' + title + ' ' + name + '!')
   
# call `greet` with a name and title:
greet('Feynman', 'Dr.')
```

## The Return Statement

Functions can return (i.e. output) values using the `return` statement. This is useful when we want to assign the result of some computation to a variable or use it in an expression. For example:

```{code-cell} ipython3
# create a function to add two numbers:
def add_numbers(a,b):
    return a + b

# store the returned value of a + b in `result`:
result = add_numbers(3,5)

print(result)
```

It is also possible to return more than one value from a function. To do this, use a comma separated list of values after the `return` statement. This will pack the returned value into a tuple of the appropriate size and return the tuple. Note that you can unpack the returned results by assigning the result to a comma separated list of variables.

To illustrate this, Let's write a Python function that solves for the roots of a quadratic equation of the form 

$$f(x) = ax^2 + bx + c$$

```{code-cell} ipython3
# create a function that solves the quadratic equation:
def solve_quadratic(a,b,c):
    
    # compute x +/- solutions with quadratic equation:
    x_plus = (-b + (b**2 - 4*a*c)**(1/2)) / (2*a)
    x_minus = (-b - (b**2 - 4*a*c)**(1/2)) / (2*a)

    return x_plus, x_minus

# call quadratic solver (store result as a tuple):
result = solve_quadratic(1,2,1)
print(result)

# call quadratic solver (unpack result into variables):
x1, x2 = solve_quadratic(1,2,1)
print(x1)
print(x2)
```

## Function Documentation

An important part of good programming practice is writing comments that document what your code does and how it works. Documentation is especially important when working with Python functions, since someone else (including you at a later time) may want to be able to use your code, but not have to understand all of the details about how your code executes. This is why functions are powerful; they allow you to write code at a higher level of abstraction than basic Python operations by reducing many lines of code to a single function call. However, with this power comes the responsibility of communicating what a function does and what kind of guarantees are provided with regards to parameters, returned values, and any other variables that my be modified during a function call. 

Up until now, we have been documenting code using single line comments (i.e. `#`). In Python, it is considered better practice to use a multi-line string called a _docstring_ instead of a single line comment to document a function. A docstring is a comment enclosed by triple quotes (`"""` ... `"""`). If an indented docstring is put beneath a function's  `def` statement, the docstring will be printed when the `help` function is used to print out the details of a function. Below, we give some examples of docstrings:

```{code-cell} ipython3
:tags: [hide-output]

# Short docstring for a function:
def greet(name):
    """ prints a greeting for the given name """
    print('Hello, ' + name + '!')


# More detailed docstring for a function:
def solve_quadratic(a,b,c):
    """
        Solves for the two roots of a quadratic equation (i.e. f(x) = ax^2 + bx + c).

        Args:
            a  (float): degree 2 coeffiient
            b: (float): degree 1 coefficient
            c: (float): degree 0 coefficient

        Returns:
            x_plus, x_minus
            
            where x_plus, x_minus may be of the int, float, or complex type. 
            In the case where f(x) has a double root, x_plus will equal x_minus.
    """

    # compute x +/- solutions with quadratic equation:
    x_plus = (-b + (b**2 - 4*a*c)**(1/2)) / (2*a)
    x_minus = (-b - (b**2 - 4*a*c)**(1/2)) / (2*a)

    return x_plus, x_minus

# print documentation for `greet`:
help(greet)

# print documentation for `solve_quadratic`:
help(solve_quadratic)
```

If you ever encounter a function that you haven't used before, you can type `help(<function>)` into your Python terminal to learn more about it.

:::{seealso}
To learn more about best practices when it comes to writing docstrings, see [PEP 257](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), which provides some guidelines for how to document parameters, returned values, etc.
:::

## Default Arguments

Sometimes, we might want the ability to specify default or recommended values for function parameters. To do this, Python supports the use of _default arguments_ in functions, which are values that are used when the corresponding arguments are not provided during function calls. Default arguments provide flexibility by allowing functions to be called with fewer arguments or with specific values for only some of the parameters.

Default arguments are defined in the function's parameter list by assigning a default value to a parameter with the `=` operator. When defining a function, parameters with default values must be placed after parameters without default values. For example:

```{code-cell} ipython3
def greet(name, message="Hello"):
    """ Prints a greeting with a name and a message """
    print(message + ', ' + name + '!')

# call greet with the default message:
greet('Albert')

# call greet with a non-default message:
greet('Albert', 'Salutations')
```

If a function has multiple default values, we might encounter some difficulties in function calls where we want to use the default value for some parameters but not others. To illustrate this, let's  add the following default arguments to out `solve_quadratic` function:

```{code-cell} ipython3
def solve_quadratic(a=1,b=0,c=0):
    """ solves a quadratic equation """
    x_plus = (-b + (b**2 - 4*a*c)**(1/2)) / (2*a)
    x_minus = (-b - (b**2 - 4*a*c)**(1/2)) / (2*a)

    return x_plus, x_minus
```

Suppose we want to solve for the roots of $2x^2 + bx - 8$, where $b$ is given the default value. If we call `solve_quadratic(2,-8)`, it will assign the value of $-8$ to $b$, not $c$, which is not what we want. To resolve this issue, we can assign each parameter by name in the function call using the `=` operator. For example:

```{code-cell} ipython3
# assigns a=2 and b=-8 (not what we wanted):
print(solve_quadratic(2,-8))

# assigns a=2 and c=-8 (what we wanted):
print(solve_quadratic(a=2,c=-8))

# assigns a=2 (resolved by order) and c=-8:
print(solve_quadratic(2,c=-8))

# If parameters are assigned by name, the order of
# parameters in the function call doesn't matter:
print(solve_quadratic(c=-8, a=2))
```

When parameters are assigned by name, the order of named parameters does not matter. However, any unnamed parameters must precede named parameters.

## Exercises

:::{dropdown} Exercise 1: List Statistics
Write a Python function called `list_stats` that takes list as a parameter and returns the minimum, mean, and maximum value in the list. 

Make sure your function includes a docstring. Test your function by calling it with a large list.

---
_Hint_: To compute the minimum and maximum of a list, you may find it helpful to use Python's `min`, `max`, and `sum` functions:
```
my_list = [1,2,3]

min(my_list) # returns 1
max(my_list) # returns 3
sum(my_list) # returns 6
```
:::

:::{dropdown} Exercise 2: Star Rectangle
Write a Python function called `print_rectangle` that prints out a rectangle consisting of `*` characters.

This function should have two parameters, `width` and `height`, indicating the width and height of the rectangle to be printed.
You function should assign a default value of 16 to `width` and 4 to `height`.

For example, the result of calling `print_rectangle(height=2)` should be:
```
****************
****************
```
:::

### Solutions

#### Exercise 1: List Statistics:

```{code-cell} ipython3
:tags: [hide-cell]

def list_stats(my_list):
    """ computes the min, mean, and max of a list """
    # compute list min, mean, and max:
    list_min = min(my_list)
    list_mean = sum(my_list)/len(my_list)
    list_max = max(my_list)

    # return list min, mean, and max:
    return list_min, list_mean, list_max

# construct a large list:
large_list = list(range(1001))

# call list_stats on large list:
print(list_stats(large_list))
```

#### Exercise 2: Star Rectangle

```{code-cell} ipython3
:tags: [hide-cell]

def print_rectangle(width=16, height=4):
    """ 
    prints a rectangle of '*' characters with 
    given width and height 
    """
    
    for i in range(height):
        print('*' * width)

# example of a function call:
print_rectangle(height=2)
```
