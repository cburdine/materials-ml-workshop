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

Scientific computing often involves using domain-specific computational tools and techniques to solve certain mathematical problems. The SciPy package in Python is a powerful library specifically designed with scientific computing in mind. It builds upon the foundation provided by NumPy and offers additional functionality for solving some of the most common problems in various scientific domains, such as physics, statistics, and engineering.

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

# Scientific Constants

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

# Integration

As we have shown previously, Scipy can also numerically integrate Python functions in the `scipy.integrate` subpackage. Integration is often necessary when computing various physical quantities, such as wave function inner products in quantum mechanics. Integration is also used to simulate the trajectories of physical systems with respect to time. For example, we can numerically evaluate the Gaussian integral

$$\int_{-\infty}^\infty e^{-x^2}\ dx = \sqrt{\pi}$$

using the following code:

```{code-cell}
import numpy as np
from scipy.integrate import quad

# Gaussian function:
def gauss(x):
    return np.exp(-x**2)

# integrate from -10^3 to +10^3 (close enough to infinity):
integral, error = quad(gauss, -1e3, 1e3)

# compare numerical result with actual result:
print(integral)
print(np.sqrt(np.pi))
```


Scipy's integration subpackage also provides some functionality for numerically solving ordinary differential equations (ODEs) with the [`scipy.integrate.odeint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html#scipy-integrate-odeint) function.

:::{seealso}
Scipy is only capable of numerical integration. For symbolic integration problems (i.e. where a closed-form mathematical expression is the desired result), take a look at the [SymPy package](https://www.sympy.org/en/index.html). Sympy can serve as a free Python-compatible alternative to other symbolic software packages, such as [Mathematica](https://www.wolfram.com/mathematica/).
:::

# Optimization

In the `scipy.optimize` subpackage, Scipy provides functionality for optimization (i.e. numerically solving for the minimum or maximum of a function) and curve fitting (fitting a curve function to data). Below, we give an example of using [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit) to fit a linear model to data:

```{code-cell}
import numpy as np
from scipy.optimize import curve_fit

# seed random number generator:
np.random.seed(0)

# generate N datapoints that fit y = 10x + 3:
N = 100
x_data = np.linspace(0,10,N)
y_noise = np.random.normal(size=N) 
y_data = 10*x_data + 3 + y_noise

# linear model of the form y = ax + b:
def linear_model(x, a, b):
    return a*x + b

# fit the linear model to data:
initial_guess = (0,0)
p_opt, p_cov = curve_fit(linear_model,x_data, y_data, initial_guess)

# print the estimated a and b coefficients:
print('Estimated a:', p_opt[0], '(should be close to 10.0)')
print('Estimated b:', p_opt[1], '(should be close to 3.0)')
```

:::{important}
When using `scipy.optimize.curve_fit`, the curve function to be fitted will be called repeatedly with the entire `x_data` array as the first argument, not with each element in `x_data` individually. Because of this, the function should return an array with the same shape as `y_data`. 
:::

## Special Functions

Finally, Scipy has support for evaluating common mathematical functions that do not admit a closed form. These functions can be found in the [`scipy.special`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit) subpackage. A few notable functions in this subpackage include:

* Airy functions: `scipy.special.airy`
* Bessel functions: `scipy.special.jv`
* Spherical Bessel functions: `scipy.special.yn`
* Gamma function: `scipy.special.gamma`
* Rieman zeta function: `scipy.special.zeta`

## Exercises

:::{dropdown} Exercise 1: Resistivity of Metals

The [_resistivity_](https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity) of a material (denoted by $\rho$) measures how strongly it resists electric current. In metals, $\rho$ typically grows as $\rho \sim T$ at high temperatures and as $\rho \sim T^n$ at low temperatures, where the degree $n$ dependins on what kind of electron interactions are dominant. Specifically, we can model $\rho$ at a temperature $T$ in Kelvin using the [Bloch-Gruneisen model](https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19334080504):

$$\rho(T) \approx \rho(0) + A\left( \frac{T}{\Theta} \right)^n \int_0^{\Theta/T} \frac{x^n}{(e^x - 1)(1 - e^{-x})}\ dx$$

where $\rho(0)$ is the residual resistivity, $A$ is a proportionality constant describing electron velocity at the Fermi surface, and $\Theta$ is the [Debye temperature](https://en.wikipedia.org/wiki/Debye_model#Debye_temperature_table) of the material. The value of the degree $n$ depends on which kind of scattering contributes most to the resistivity:

* $n = 2$ implies that the resistance is due to dominance of electron-electron scattering.
* $n = 5$ implies that the resistance is due to dominance of electron-phonon scattering.

Write a Python function that numerically estimates $\rho(T)$ using the Bloch-Gruneisen model. Your function should have the following signature:

```
def rho_estimate(T,rho_0, A, theta, n):
    ...
```

Note that you may need to define another function to seve as the integrand in the Bloch-Gruneisen model, which you can pass into `scipy.integrate.quad` to integrate.
:::

:::{dropdown} Exercise 2: Modeling the Resistivity of Platinum
    
Let's try to fit the `rho_estimate` function from Exercise 1 to some experimental resistivity data gathered on a Platinum sample. Platinum has high electron-phonon coupling, so we will assume a degree $n=5$ trend of $\rho$ versus $T$. The resistivity data is given below, in a Python dictionary format:

```
PT_RHO_DATA = {
# Temp. (K) : Resistivity (Ω)
    14.     :   1.797,
    20.     :   2.147,
    30.     :   3.508,
    40.     :   5.938,
    50.     :   9.228,
    70.     :  17.128,
    100.0   :  29.987,
    200.0   :  71.073,
    300.0   : 110.450,
    400.0   : 148.620,
    1000.0  : 353.402
}
```

Using the `scipy.optimize.curve_fit` function, fit this data to $n=5$ Bloch-Gruneisen model to estimate the value of $A$ and $\rho(0)$. You may need to look up the value of the Debye temperature of Platinum.

---
_Note_: This exercise (and the data used) was adapted from Chapter 8.6, Problem 2 in _Foundations of Solid State Physics_ by Roth and Carroll. 
:::

### Solutions

#### Exercise 1: Resistivity of Metals

```{code-cell}
:tags: [hide-cell]

def bg_integrand(x,n):
    """ The integrand of the Bloch-Gruneisen model """
    return x**n / ((np.exp(x) - 1)*(1 - np.exp(-x)))

def rho_estimate(T, rho_0, A, theta, n):
    """
    Estimates the temperature-dependant resistivity of a metal using the Bloch-Gruneisen model:

    Arguments:
        T: temperature (K)
        rho_0: residual resistance of material (Ω)
        theta: Debye temperature of material (K)
        n: degree of fit (n=2 or n=5)
    """
    integral, err = quad(bg_integrand, 0, theta/T, args=(n,))
    return rho_0 + A * (T/theta)**n * integral
```

#### Exercise 2: Modeling the Resistivity of Platinum

```{code-cell}
:tags: [hide-cell]

from scipy.optimize import curve_fit

# Given Data (Resistivity of Pt vs Temp.):
PT_RHO_DATA = {
# Temp. (K) : Resistivity (Ω)
    14.     :   1.797,
    20.     :   2.147,
    30.     :   3.508,
    40.     :   5.938,
    50.     :   9.228,
    70.     :  17.128,
    100.0   :  29.987,
    200.0   :  71.073,
    300.0   : 110.450,
    400.0   : 148.620,
    1000.0  : 353.402
}

# Debye Temerature of Platinum:
PT_DEBYE_TEMP = 240 # (K)

# convert data to numpy arrays:
data = np.array(list(PT_RHO_DATA.items()))
T_data = data[:,0]
rho_data = data[:,1]

# write a function that predicts platinum's rho(T) for an array of T values:
def pt_rho_estimate(T_array, rho_0, A):
    """ Estimates the B.G resistivity for Platinum for an array of T values"""
    return np.array([ 
        rho_estimate(T, rho_0, A, theta=PT_DEBYE_TEMP, n=5)
        for T in T_array
    ])

# Determine the optimal parameters:
p_opt, p_cov = curve_fit((pt_rho_estimate), 
                         T_data,
                         rho_data,
                         p0=(1, 300),
                         bounds=([0,0],[np.min(rho_data),1000]))

print('Estimated rho_0:', p_opt[0])
print('Estimated A:', p_opt[1])
```
