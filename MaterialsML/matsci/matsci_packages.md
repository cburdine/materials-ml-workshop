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

# Materials Science Python Packages

There are several packages that may be useful in studying materials science. We'll briefly introduce a few of them:
1. The Atomic Simulation Environment (ASE)
2. Python Materials Genomics (PyMatGen)
3. The Materials Project API (MPRester)

## ASE - The Atomic Simulation Environment

![](ase256.png)

The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) is an open-source set of tools and Python modules for setting up, manipulating, running, visualizing and analyzing atomistic simulations. ASE can help you molecules and crystals, and then simulate them at different levels of theory (density functional theory, molecular dynamics, etc.). ASE can interface with a variety of simulation software platforms including [VASP](https://www.vasp.at), [Quantum ESPRESSO](https://www.quantum-espresso.org), [Q-Chem](https://www.q-chem.com), [Gaussian](https://gaussian.com), and others (see the [full list](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#supported-calculators)) through tools called calculators. ASE can create input files, launch simulations, and parse the output.

### Installation
To use ASE, you must first install the `ase` Python module. You may use a command such as `pip3 install ase` to do this.

### Usage

To simulate a material or molecule in ASE, the workflow is typically as follows:
1. Build an ASE `Atoms` object to represent your molecule or material
2. Use an ASE `calculator` object to perform a simulation and parse/visualize the results.

:::{Note}

Performing atomistic simulations are beyond the scope of this discussion. We will show you how to create an `Atoms` object. More information about performing simulations using `Atoms` objects and `calculator` objects may be found on the ASE documentation page [Atoms and Calculators](https://wiki.fysik.dtu.dk/ase/gettingstarted/tut01_molecule/molecule.html).
:::

### `ase.Atoms` Objects
In `ase`, we use an `Atoms` object for an atomistic description of a material system. An `Atoms` object is actually a collection of `Atom` objects, each of which describes an atom, with member data such as `symbol` (string), `position` (a 3-element tuple of Cartesian coordinates, in Angstroms). Other atomic properties could be specified, such as `mass`, `charge`, etc.

### Building Simple Molecules

Python code that uses `ase` must include an `import` statement that imports the `ase` tools (functions or classes) you want to use. Here, we will:
* use the `ase.build.molecule()` function to construct a water molecule, and
* make a representation of the molecule using the `ase.visualize.view()`

```{code-cell}

from ase.build import molecule
from ase.visualize import view

"""
   Build an ase.Atoms object to represent a water molecule.
   We use the molecule() function to do this, and we specify
   the chemical formula for water.
"""
water = molecule('H2O')

# The view() function provides an interactive, 3D visualization
view(water, viewer='x3d')
```

The `molecule()` function is provided as a simple way to build an `Atoms` object. Here, a molecule is specified using a Python string containing a chemical formula, and only a very limited set of molecules are supported. The list of available molecules is found in the `ase.collections.g2.names` list:
```{code-cell}

from ase.collections import g2

# print the ase.collections.g2.names list
print(g2.names)
```

Let's do this again for a formic acid molecule. We will also have Python print the coordinates of each atom.

```{code-cell}

from ase.build import molecule
from ase.visualize import view

# construct a formic acid molecule
atoms = molecule('HCOOH')

"""
   Let's also print the symbol and coordinates of each atom.
"""
print('Show atomic coordinates:\n')
for X in atoms:
    x, y, z = X.position
    print('{0}\n  x: {1} Ang.\n  y: {2} Ang.\n  z: {3} Ang.'.format(X.symbol,
                                                                    x, y, z)
                                                                    )

# 3D visualization
view(atoms, viewer='x3d')

```

Having constructed an `ase.Atoms` object to represent water molecule, we could create a `calculator` object to run a simulation.

### Building Complex molecules

If you want to go beyond the simple molecules ASE can create using the `moleucles()` function, you may use other strategies:
* Construct a molecule from a structure file (`*.cif`, `*.xyz`, etc.)
* Read the structure from simulation output
* Obtain a structure from a molecular database

To read a structure from a file, use the `read()` function from the `ase.io` module. This module allows ASE to read from and write to files containing information about materials ([documentation](https://wiki.fysik.dtu.dk/ase/ase/io/io.html)).

### Building Simple Crystals

## PyMatGen - Python Materials Genomics

![](pymatgen.svg)

## MPRester - The Materials Project API

[The Materials Project API](https://materialsproject.org/api) allows a user to query information from [the Materials Project](https://materialsproject.org).

### Installation

To use ASE, you must first install the `ase` Python module. You may use a command such as `pip3 install ase` to do this.

### An API Key is Required for Usage

```{code-cell}

from ase.build import bulk
from ase.visualize import view

atoms = bulk('C','diamond', 3.57)

# This is an added comment
view(atoms, viewer='x3d')
```
