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

# Python Functions and Classes

Functions are an essential concept in Python and programming in general. They allow you to encapsulate reusable pieces of code into named blocks, which can be called and executed whenever needed. Functions promote code modularity, readability, and reusability, making them a fundamental building block for structuring and organizing programs. Likewise, Classes in Python provide a way of encapsulating both data and the operations performable on it. In this section we will focus on writing Python functions, and see how they can help us write more concise code.

## Function Basics

In mathematics, functions are transformations that map inputs to outputs. The same is true for Python, except for the fact that Python functions can also change the values of variables, modify mutable data types (like lists), etc. So far, we have encountered several of Python's built-in functions, for example:

```{code-cell}
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

```{code-cell}
# create a function to print out a greeting:
def greet():
    print('Hello, world!')
```

In the example, we have defined a function `greet` that does not take any parameters (similar to the `list.pop`) function above. To execute a function and perform its associated actions, you call the function by using its name followed by parentheses `()`. If the functon has parameters, you can pass values within the parentheses. Since our `greet` function has no parameters, we can call the function (i.e. execute is corresponding block of code) by using its name followed by parentheses `()`:

```{code-cell}
greet()
```
Upon calling the `greet` function, we see that the `print('Hello, world!')` line is executed, resulting in the output `Hello, world!`.

## Function Parameters: 

Like mathematical functions, Python functions can accept _parameters_, which are variables used to pass values into the function. Parameters are specified within the parentheses of the function definition. To call a function with parameters, we simply put the value of the parameter between the parentheses when the function is called. 

For example, let's re-write our `greet` function to accept a parameter called `name`, and then call the function with the value `'Albert'` as `name`:

```{code-cell}
# updated function that prints a greeting for a specific name:
def greet(name):
    print('Hello, ' + name + '!')

# call greet with 'Albert':
greet('Albert')
```

Functions can also have multiple parameters, which we can denote using a comma separated list of parameters. When calling a function witih multiple parameters, be sure that the order of the parameters match the order of the corresponding values:
```{code-cell}
# This function prints out a greeting of a name (with a title):
def greet(name, title):
    print('Hello, ' + title + ' ' + name + '!')
   
# call `greet` with a name and title:
greet('Feynman', 'Dr.')
```

## The Return Statement

Functions can return (i.e. output) values using the `return` statement. This is useful when we want to assign the result of some computation to a variable or use it in an expression. For example:

```{code-cell}
# create a function to add two numbers:
def add_numbers(a,b):
    return a + b

# store the returned value of a + b in `result`:
result = add_numbers(3,5)

print(result)
```

Sometimes, it might be necessary to return more than one value. To return multiple values, use a comma separated list of values after the `return` statement. This will pack the returned value into a tuple of the appropriate size and return the tuple. Note that you can upack the returned results by assigning the result to a comma separated list of variables.

To illustrate this, Let's write a Python function that solves for the roots of a quadratic equation of the form 

$$f(x) = ax^2 + bx + c$$

```{code-cell}
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

Up until now, we have been documenting code using single line comments (i.e. `#`). In Python, it is considered better practice to use a multiline string called a _docstring_ instead of a single line comment to document a function. A docstring is a comment enclosed by triple quotes (`"""` ... `"""`). If an indented docstring is put beneath a function's  `def` statement, the docstring will be printed when the `help` function is used to print out the details of a function. Below, we give some examples of docstrings:

```{code-cell}
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

:::{seealso}
To learn more about best practices when it comes to writing docstrings, see [PEP 257](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), which provides some guidelines for how to document parameters, returned values, etc.
:::

## Default Arguments


