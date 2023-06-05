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

# execute while loop:
while step < 5:

    # print step:
    print('step:', step)
    
    # increment step:
    step = step + 1

# print "Done" upon completion of the loop:
print('Done')
```

Sometimes, we might want to exit a `while` loop from the middle of the loop body. This can be achieved with the `break` statement, which immediately exits the innermost loop:

```{code-cell}
# initialize step counter:
step = 0

while step < 1000:
    
    # begin loop body:
    print('Starting step', step)

    # condition for breaking the while loop:
    if step == 5:
        break
    
    
    # end loop body:
    print('Ending step', step)
     
    # increment step counter (shorthand for 'step = step + 1'):
    step += 1
