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

# Unsupervised Learning

Now that we have wrapped our discussion of supervised learning, let's begin exploring _unsupervised learning_. Unlike supervised learning problems, which are concerned with making predictions based on labeled data, unsupervised learning problems are concerned with the identification of trends, patterns, and clusters based on unlabeled data. In supervised learning, the dataset consists of $(\mathbf{x},y)$ pairs; however in unsupervised learning, we only have the raw datapoints $\mathbf{x}$ to work with.


As one might expect, unsupervised learning problems are generally more difficult than supervised problems. In supervised learning, where used loss functions $\mathcal{E}(f)$ in combination with train, validation, and test sets to quantitatively measure the accuracy of proposed model. However, in unsupervised learning, there is often no clear metric that can be used to gauge the accuracy of any trends, patterns or clusters that are identified in a dataset. Often, unsupervised learning must be guided by expert intuition. As materials scientists and researchers, we supply this intuition by consulting the literature and applying known theoretical models to explain the data. When analyzing data, often we know what kinds of trends, patterns and clusters we expect to see, and we use our expert judgement to determine what kind of unsupervised learning methods are most applicable to our data.

## Unsupervised Learning Methods

![Unsupervised Learning Problems](unsupervised_learning.svg)

## Dimensionality and Correlation

(TODO)

## The Curse of Dimensionality

(TODO)
