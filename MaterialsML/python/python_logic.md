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

# Logic and Flow Control

An important part of programming is the use of logic to control the execution of a Python program. Sometimes, we may only want to do certain computational steps when certain conditions are met. For example, in performing a division operation (i.e. `z = x / y` for two variables `x` and `y`) we may want to have a special procedure in place to handle when `y` is equal to 0. This is where logic and conditional statements come in.

## The Boolean Type

In order to understand how Python handles logic, we must first learn about the Boolean type, which is called `bool` in Python. A Boolean type (named after the mathematician [George Bool](https://en.wikipedia.org/wiki/George_Boole)) is a value that can store either a `True` or `False` value. Let's create some `bool` varibles using the following Python code:

```{code-cell}
# True and False are constant Boolean values:
print(type(True), type(False))

# create some Boolean variables
bool_a = True
bool_b = False

# print out the type of these variables:
print(bool_a, bool_b)
print(type(bool_a), type(bool_b))
```

Much like how we can perform arithmetic on `int` and `float` types, we can perform the `and`, `or`, and `not` operations from [Boolean algebra](https://en.wikipedia.org/wiki/Boolean_algebra#Operations) on Python `bool` types:
```{code-cell}
# behavior of the `and` operator:
print('"and" operator:')
print(False and False)
print(False and True)
print(True and False)
print(True and True)

# behavior of the `or` operator:
print('"or" operator:')
print(False or False)
print(False or True)
print(True or False)
print(True or True)

# behavior of the `not` operator:
print('"not" operator:')
print(not False)
print(not True)
```

In Python, the order of operations for `bool` operators is `not`, followed by `and`, followed by `or`. However, it is generally a good idea to use parentheses to avoid any ambiguity when combining multiple Boolean operations:

```{code-cell}
# create boolean variables:
bool_a = True
bool_b = False
bool_c = False

# evaluated nested expression:
print((not bool_a) and ((not bool_b) or bool_c))
```

## Comparison Operators

Sometimes, we need to compare one or more numeric values in Python. This can be achieved with comparison operators. For two Python variables `a` and `b` of the `int` or `float` type,  we can perform the following comparisons:

* Less than ($a < b$): `a < b`
* Less than or equal to ($a \le b$): `a <= b`
* Greater than ($a > b$): `a > b`
* Greater than or equal to: ($a \ge b$): `a >= b`
* Equal to: ($a = b$): `a == b`
* Not equal to: ($a \neq b$): `a != b`

Since a comparison naturally evaluates to be either true or false, the result of applying a comparison operation in Python is a `bool` type:

```{code-cell}
# create some numerical variables:
x = 0.25
y = -3
z = 20
z_2 = 20.0

# simple comparisons:
print('Some simple comparisons:')
print(x < y)
print(y < z)
print(x == y)
print(z >= z_2)

# more comparisons:
print('More comparisons:')
print(z == z_2)
print(z != z_2)
print(x == 0)
print(y >= 3.0)
```

Sometimes it is necessary to combine together comparison operators with Boolean operators:

```{code-cell}
# create som numerical variables:
x = 0.25
y = 1.23
z = 0.0

# check if x is outside the interval (0,1]:
print((x <= 0) or (1 < x))

# check if x is inside the interval (0,1]:
print((0 < x) and (x <= 1))

# This is equivalent to the previous expression:
print(0 < x <= 1)
```

We can also apply conditional operators to variables of the `str` (string) type, which store text. For any two strings `str_a` and `str_b`, the `<` operator evaluates to True if `str_a` preceeds `str_b` alphabetically:

```{code-cell}
# create some strings:
str_a = 'alpha'
str_b = 'beta'
str_c = 'alpha2'
str_d = 'alpha'

# compare strings:
print(str_a < str_b)
print(str_b <= str_a)
print(str_c <= str_a)
print(str_a == str_d)
```

:::{note}
Technically, strings in Python are compared character-by-character according to each character's [ASCII character code](https://www.asciitable.com/). This means that the string `'BBB'` precedes the string `'aaa'` according to Python's `<` operator, since capital letters precede lowercase letters in ASCII. In order to compare strings in a true case-insensitive lexicographical ordering, you can use the `str.lower()` function:
```
str_a.lower() < str_b.lower()
```
:::

## Conditional statements

Sometimes, it is necessary to control whether or not a block of code executes based on a boolean condition. For example, to avoid division by $0$, we might want to check if the demoninator in a division operation is zero prior to performing the division. We can accomplish this in Python using an `if` statement. An `if` statement executes only if the subsequent Boolean value (or expression that evaluates to a boolean value) is `True`. To see how `if` statements work, try executing the following Python code: 

```{code-cell}
# initialize variables:
numerator = 10.0
denominator = 0.0

# initalize the quotient:
quotient = 0.0

# perform division only if the denominator is nonzero:
if denominator != 0:
    quotient = numerator / denominator

# print the result:
print('Quotient:', quotient)
```
We can see from the output that the line `quotient = numerator / denominator` is not executed, since the condition `denominator != 0` evaluates to `False`. Also note that the body of the `if` statement is indented by either a single tab or four spaces. 

:::{important}
Although Python is generally not very strict about whitespace between operations, it is important to make sure that the beginning of each indented line in an `if` statement has the proper amount of whitespace. This whitespace can be either four spaces or a single tab, but whether tabs or spaces are used must be consistent throughout the entire block of Python code. If you are writing Python code in an editor or Integrated Development Environment (IDE), it is usually good practice to configure your editor to convert tabs to spaces when saving.
:::

Sometimes, we might also want to handle the case when the expression inside an `if` statement evaluates to `False`. This is achieved through `if`/`else` statements:

```{code-cell}
# intialize some names:
name_1 = 'John von Neumann'
name_2 = 'Paul Erdos'

if name_1 < name_2:
    print(name_1, 'precedes', name_2)
else:
    print(name_2, 'precedes', name_1)
```

We can also chain multiple mutually exclusive conditions into `if`/`elif`/`else` statements (the `elif` is a contraction of "else if"). In thes statments, conditions are checked from top to bottom, and the block of the first condition that is met is executed. If an `else` block is specified, that block is executed only if none of the preceding conditions are met. For example: 

```{code-cell}
# initialize a numeric value:
value = 100.0

# print out the sign of x:
if value < 0:
    print('Value is negative.')
elif value == 0:
    print('Value is zero.')
else:
    print('Value is positive.')
```

We can also nest conditional statements by applying double indententation to the body of the interior conditional statement:

```{code-cell}
# initialize x
x = 0.0

# check if x lies inside [-1,1]
if -1 < x < 1:

    # check if x lies at the origin:
    if x == 0:
        print('x lies at the origin')
     
    print('x lies within [-1,1]')

```

## Exercises

:::{dropdown} Exercise 1: Classifying Chemical Compounds
Write a program that uses conditional statements to classify chemical compounds as organic, carbon-based, or inorganic based upon the following (naive) rules:

1. If a compound contains carbon ('C') and hydrogen ('H') in its chemical formula, it is organic.
2. If a compond contains carbon but no hydrogen in its chemical formula, it is carbon-based.
3. If a compond contains neither carbon nor hydrogen in its chemical formula, it is inorganic.

The program should take a compound's formula string (i.e. `'CH3COOH'`, `'NH3'`, or `'SiC'`) that is stored in a variable `formula` and print out the classification of the compound. Try to make your logic and conditional statements as concise as possible.

---
_Hint_: To test if a string contains a character or another string as a substring, you can use the `in` operator. For example:

```
string_1 = 'Hello World.'
string_2 = 'I love Chemistry!'

print('H' in string_1) # True
print('C' in string_1) # False
print('C' in string_2) # True
```
:::

### Solutions:

## Exercise 1: Classifying Chemical Compounds

```{code-cell}
:tags: [hide-cell]
# initialize formula:
formula = 'C8H10N4O2'

# check if formula contains carbon:
if 'C' in formula:

    # check if formula also contains hydrogen:
    if 'H' in formula:
        print('Organic')
    else:
        print('Carbon-based')
else:
    print('Inorganic')
```
