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

# Loops

In Python, loops are one of the most important control structures. They are used to repeatedly execute a block of code until a certain condition is met.

## The While Loop

A `while` loop is used to repeatedly execute a block of code as long as a certain condition is true. The general syntax of a `while` loop in Python is as follows:

```
while condition:
    # code to be executed as long as the loop condition is met
```

Here's a brief summary of how a while loop works:

* First, the loop starts by checking the condition. If the condition is true, the code block within the loop is executed. If the condition is false, the loop is skipped, and the program execution moves to the next statement after the loop.

* After executing the code block, the condition is checked again. If it is still true, the code block is executed again. This process continues until the condition becomes false.


```{code-cell}
# initialize a step counter:
step = 0

# execute the `while` loop:
while step < 5:

    # print step:
    print('step:', step)
    
    # increment step:
    step = step + 1

# print "Done" upon completion of the loop:
print('Done')
```

## The For Loop

Keeping track of a loop count variable and incrementing the variable in the loop body can be tedious. This is why Python has the `for` loop, which provides a concise way of iterating over sequential values. The general syntax of a `for` loop is as follows:

```
for item in sequence:
    # code to be executed for each item in the sequence
```

The execution of `for` loops is similar to `while` loops. Here's a breakdown of how a for loop works:

* The loop starts by initializing a variable (`item` in the above example) that will take on the value of each item in the sequence one by one.

* The loop then iterates over the sequence, executing the code block within the loop for each item. Once the code block is executed for the current item, the loop moves to the next item in the sequence.

* The loop continues this process until all the items in the sequence have been processed, after which it terminates and the program execution moves to the next statement after the loop.

To iterate over a sequence of consecutive integers, we can use the `range` function. For a Python integer `n`, the sequence `range(n)` consists of all integers 0,1,2,...,n-1, not including the value of `n` itself. The sequence `range(m,n)` behaves similarly, but starts at the value of `m` instead of 0.

We can use the `range` sequence in a for loop as follows:

```{code-cell}
# iterate over i = 1,2,...,4
for i in range(5):
    print('value of i:',i)

# iterate over j = 7,8,9
for j in range(7,10):
    print('value of j:',j)
```

## Iterating over Lists:

We can also use `for` loops for iterating over lists of values. In Python, a list is denoted by a sequence of comma-separated values within square brackets (`[`...`]`). We will discuss lists in greater detail in the next section. For now, just keep in mind that we can interate over a finite sequence of values as follows:

```{code-cell}
for element in ['H', 'He', 'Li', 'Be']:
    print(element)
```

Sometimes, we need to iterate over both the indices and values of a list simultaneously. This can be done using the `enumerate` function:

```{code-cell}
# initialize elements in a list:
element_list = ['H', 'He', 'Li', 'Be']

# enumerate each element and its index:
for i, element in enumerate(element_list):
    print('Index:', i)
    print('Element:', element)
```

:::{note}
Unlike other languages such as MATLAB or the Wolfram Language, sequences in Python are indexed starting at 0, not 1. This is why the `range` and `enumerate` functions start at 0 and end at n-1. We will see more of this in the next section when we discuss Python lists.
:::

## Example: Computing a factorial

Let's apply what we have learned about loops to compute the factorial function. The factorial of a number $n$ is the product:

$$n! = n(n-1)(n-2)...(2)(1)$$

```{code-cell}
# number to compute factorial of:
n = 30

# iterate over 1,2,...,n and compute the product:
product = 1
for i in range(1,n+1):
    product *= i

# print the resulting factorial product:
print('n! =', product)
```

## Exercises:

::: {dropdown} Exercise 1: Fibbonacci Sequence

Using a `for` loop, write some Python code to compute the $n$th number in the [Fibbonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_sequence) $F_n$, which satisfies the recurrence relation:

$$F_n = F_{n-1} + F_{n-2}$$

with $F_{1} = 1$ and $F_{2} = 1$.

Print out the value of $F_{100}$ and the quotient $F_{101} / F_{100}$. 

Does this quotient [look familiar](https://en.wikipedia.org/wiki/Golden_ratio)?
:::

:::{dropdown} Exercise 2: Total Word and Character Count

When writing essays, papers, abstracts, etc., sometimes we must make sure that our papers are within recommended word and character counts. Let's write a Python program that counts the number of words and characters in a string variable called `essay`. We can convert `essay` into a list of words using the following code:

```
# initialize essay as a string:
essay = 'This is an essay.'

# split text into a list of words:
word_list = essay.split()
```
Using a `for` loop, iterate over all of the words in `word_list` and count the total number of words and word characters in the essay.

---

_Hint_: To find the number of characters in a string, use the `len` function:

```
len('Hello') # <-- evaluates to 5
```
:::

### Solutions:

#### Exercise 1: Fibbonacci Sequence

```{code-cell}
:tags: [hide-cell]

n = 100

f_prev = 0
f_current = 1

for i in range(n):
    f_next = f_prev + f_current
    f_prev = f_current
    f_current  = f_next

phi = f_current / f_prev

print('F(n):', f_prev)
print('F(n+1)/F(n):', phi)
```

#### Exercise 2: Total Word and Character Count

```{code-cell}
:tags: [hide-cell]
# initialize essay as a string:
essay = """
Once upon a time and a very good time it was 
there was a moocow coming down along the road 
and this moocow that was coming down along the 
road met a nicens little boy named baby tuckoo.
"""

# split text into a list of words:
word_list = essay.split()

# initialize word and character counts:
character_count = 0
word_count = 0

# count words and characters in word list:
for word in word_list:
    character_count += len(word)
    word_count += 1

print('Word count:', word_count)
print('Character count:', character_count)
```
