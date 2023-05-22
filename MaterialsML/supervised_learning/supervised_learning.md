# Supervised Learning

Now that we have reviewed some of the necessary background material, we will will begin examing the most common types of machine learning problem: _supervised learning_. Supervised learning is used for problems where the available data contains many different labeled examples, and the problem involves finding a model that maps a set of features (inputs) to labels (outputs).

Although data can take many different forms (i.e. numbers, vectors, images, text, 3D structures, etc.), we can think of a dataset as a set of of $(\mathbf{x}, y)$ pairs, where $\mathbf{x}$ is a set of features, and $y$ is the label to be predicted. For now, we will consider the simplest case where $\mathbf{x}$ is a vector or floating-point numbers and $y$ is either (a) one of a finite number of mutually classes, or (b) a scalar quantity. In case (a), the supervised learning problem is a _classification problem_, where we must learn a model that makes prediction $\hat{y}$ of the class $y$ associated with the set of features $\mathbf{x}$. In case (b), the supervised learning problem is a _regression problem_, where we must learn a model that produces an estimate $\hat{y}$ of $y$ based on $\mathbf{x}$.

For both classification and regression problems, we can think of a model as function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that maps the space of possible inputs $\mathcal{X}$ into the space of possible outputs $\mathcal{Y}$:

![Illustration of a Supervised Model](supervised_model.svg)



## Model Selection

```{epigraph}

"Entities must not be multiplied beyond necessity."

-- Occam's Razor
```

## Acquiring Data

(TODO)

## Handling Data

(TODO)

### The Training Set

(TODO)

### The Validation Set

(TODO)

### The Test Set

(TODO)

### Normalizing Data

(TODO)

## Key Steps of Supervised Learning

(TODO)

## Exercises

(TODO)
