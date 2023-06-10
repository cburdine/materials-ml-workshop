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
# The Scipy Package

Scientific programming involves using computational tools and techniques to solve scientific and mathematical problems. The SciPy package in Python is a powerful library specifically designed for scientific and technical computing. It builds upon the foundation provided by NumPy and offers additional functionality for a wide range of scientific computations.

SciPy consists of various modules, each focusing on specific scientific computing tasks. Some of the key modules available in SciPy include:

* `scipy.constants`: Provides physical and mathematical constants.
* `scipy.optimize`: Offers functions for optimization and root finding.
* `scipy.integrate`: Provides numerical integration functions.
* `scipy.interpolate`: Offers interpolation functions for smoothing or approximating data.
* `scipy.signal`: Provides signal processing functions.
* `scipy.linalg`: Offers linear algebra functions.
* `scipy.sparse`: Provides tools for sparse matrices and linear algebra operations.
* `scipy.stats`: Offers statistical functions and probability distributions.
* `scipy.special`: Provides special mathematical functions.

The SciPy package provides a comprehensive suite of tools and functions for various scientific computing tasks, making it a valuable resource for researchers, engineers, data scientists, and anyone involved in scientific programming. It enables efficient and reliable numerical computations, data analysis, and modeling, and helps simplify complex scientific problems by providing ready-to-use algorithms and methods.

# Scientific Units

The `scipy.constants` subpackage in SciPy provides a collection of physical and mathematical constants. These constants are useful in scientific and engineering calculations, as they allow you to reference commonly used values without the need to look them up or hard-code them into your code.

To import the constants subpackage use the import statement

```{code-cell}
from scipy import constants
```

All constants in the `scipy.constants` package are in [SI units](https://en.wikipedia.org/wiki/International_System_of_Units) and are some are written with upppercase letters. For example:

```{code-cell}
:tags: [hide-output]
print(constants.pi)             # 3.141592653589793 
print(constants.speed_of_light) # 299792458.0 [m/s]
print(constants.Avogadro)       # 6.02214076e+23 [mol^(-1)]
print(constants.G)              # Gravitational constant [m^3 / kg s^2 ] 
print(constants.Boltzmann)      # Boltzmann constant [J/K]
print(constants.m_e)            # Electron mass [kg]
```

# Integration with Scipy

As we have shown previously, Scipy also has funcioonality to integrate Python functions.

# Optimization with Scipy
