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

# Neural Networks

So far, we have encountered supervised models such as linear regression, decision trees, kernel machines, etc. While these models have the benefit of being interpretable, some of them may fail to perform well on datasets that are exceptionally large or complex (i.e. having a significant degree of non-linearity). For these large or complex datasets, only models with a significant degree of flexibility may yield the kind of discriminative capability needed to obtain high accuracy. In this section we will learn about one of the most flexible kind of non-linear model: _neural networks_.

![Simple Neural Network](./simple_nn_overview.svg)

Neural networks are a class of machine learning models inspired by the structure and functioning of neurons in the brains of living organisms. They are designed to recognize patterns and relationships within complex datasets and make predictions or decisions based on that information. At their core, neural networks consist of interconnected nodes (i.e. neurons), organized into sequential layers layers. The most common type of neural network is called a _feed-forward neural network_. It consists of an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to neurons in the subsequent layer, and these connections are associated with weights.

## General Applications of Neural Networks

Neural networks are capable of learning complex representations and capturing intricate relationships in the data. They can handle a wide variety of tasks, including image and speech recognition, natural language processing, and time series analysis. The power of neural networks lies in their ability to automatically extract meaningful features from raw data, enabling them to generalize well to unseen examples.

Over the years, neural network architectures have evolved to address specific challenges. Some notable network architectures include _convolutional neural networks_ (CNNs) for image processing, _recurrent neural networks_ (RNNs) for sequence data, and _graph neural networks_ for making predictions based on _graphs_, which are networks of data points connected by labeled edges. These specialized architectures have significantly advanced the fields of computer vision and natural language processing and have also been used to solve difficult problems in  many scientific fields.

## Neural Networks and Materials Science:

In Materials Science, neural networks a becoming an important tool used for predicting material properties in the absence of a comprehensive theory that relates a material's atomic structure to the properties its exhibits. Currently, neural networks are being employed to design novel materials with desired properties, screen large materials databases, and to provide a high-throughput alternative to computationally inefficient methods used for computing material properties, such as Density Functional Theory (DFT).

## Challenges

Neural networks have gained immense popularity due to their remarkable performance in various domains. However, they also come with challenges such as the need for large labeled datasets, computational resources, and the potential for overfitting. Nonetheless, their versatility and ability to learn from data make them a fundamental tool in modern data-driven science, especially in fields such as materials science.
