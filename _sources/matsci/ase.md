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


# ASE - The Atomic Simulation Environment

![](ase256.png)

The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) is an open-source set of tools and Python modules for setting up, manipulating, running, visualizing and analyzing atomistic simulations. ASE can help you molecules and crystals, and then simulate them at different levels of theory (density functional theory, molecular dynamics, etc.). ASE can interface with a variety of simulation software platforms including [VASP](https://www.vasp.at), [Quantum ESPRESSO](https://www.quantum-espresso.org), [Q-Chem](https://www.q-chem.com), [Gaussian](https://gaussian.com), and others (see the [full list](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#supported-calculators)) through tools called calculators. ASE can create input files, launch simulations, and parse the output.

`ase` has a vast set of modules and functions, giving it vast and powerful functionality. We will only scratch the surface in this brief introduction to `ase`.

## Installation
To use ASE, you must first install the `ase` Python module. You may use a command such as `pip3 install ase` to do this.

## Usage

To simulate a material or molecule in ASE, the workflow is typically as follows:
1. Build an ASE `Atoms` object to represent your molecule or material
2. Use an ASE `calculator` object to perform a simulation and parse/visualize the results.

:::{Note}

Performing atomistic simulations are beyond the scope of this discussion. We will show you how to create an `Atoms` object. More information about performing simulations using `Atoms` objects and `calculator` objects may be found on the ASE documentation page [Atoms and Calculators](https://wiki.fysik.dtu.dk/ase/gettingstarted/tut01_molecule/molecule.html).
:::

## `ase.Atoms` Objects
In `ase`, we use an `Atoms` object for an atomistic description of a material system. An `Atoms` object is actually a collection of `Atom` objects, each of which describes an atom, with member data such as `symbol` (string), `position` (a 3-element tuple of Cartesian coordinates, in Angstroms). Other atomic properties could be specified, such as `mass`, `charge`, etc.

## Building Simple Molecules

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

Let's do this again for a formic acid molecule. Additionally, we will also  print the x, y, and z coordinates of each atom.


```{code-cell}

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
```

We can still make an interactive, 3D visualization:

```{code-cell}

"""
   The view() function should occur last for an interactive result.
"""
# 3D visualization
view(atoms, viewer='x3d')

```

Having constructed an `ase.Atoms` object to represent water molecule, we could create a `calculator` object to run a simulation.

## Building Complex molecules

If you want to go beyond the simple molecules ASE can create using the `moleucles()` function, you may use other strategies:
* Construct a molecule from a structure file (`*.cif`, `*.xyz`, etc.)
* Read the structure from simulation output
* Obtain a structure from a molecular database

To read a structure from a file, use the `read()` function from the `ase.io` module. This module allows ASE to read from and write to files containing information about materials ([documentation](https://wiki.fysik.dtu.dk/ase/ase/io/io.html)).

## Building Simple Crystals - Bulk Silicon

We will start with a simple (bulk) silicon crystal using the `bulk()` function in the `ase.build` module.

```{code-cell}

from ase.build import bulk
from ase.io import write   # helps us save an image

atoms = bulk('Si')

"""
   This is a easy way to make a simple (static) visualization.
"""
write('silicon_basis.png', atoms, show_unit_cell=2)

```

The static image of the silicon crystal (the two-atom basis for the FCC crystal) is given below.

![](silicon_basis.png)

We can also make an interactive 3D image:

```{code-cell}

view(atoms, viewer='x3d')

```

## Building a 2D System - a MXene

The `ase.build` module provides functions for building 2D structures. For example:
* `graphene_nanoribbon()` may be used to make graphene nanoribbons and graphene sheets.
* `mx2()` may be used to build MXene and [transition metal dichalcogenide](https://www.sciencedirect.com/topics/materials-science/transition-metal-dichalcogenides#:~:text=TMD%20monolayers%20are%20structurally%20of,octahedral%20or%20trigonal%20prismatic%20coordination.) (TMD) monolayers.

Here, we will make a MXene.

```{code-cell}

from ase.build import mx2
from ase.io import write
from ase.visualize import view # 3D interactive image

# This forms a primitive unit cell
Ti2C = mx2('CTi2', vacuum = 15) # unit cell

"""
   We can also build a sheet. We form a supercell by repeating the unit cell
   3x in the x and y directions, and only one time in the z direction.
"""

# Static image of the unit cell
rotation = '0z,-60x'
write('Ti2C_unit_cell.png', Ti2C, show_unit_cell=2, rotation=rotation)

```

As static image of the unit cell is given below. Since a structure like this would likely be used in a DFT calculation, and DFT calculations often have periodic boundary conditions, the unit cell features a large air gap to keep separate the main sheet from its images in the z direction.

![](Ti2C_unit_cell.png)

Now, we form a supercell by repeating the unit cell in space. To repeat the primitive cell, described by the `Ti2C` object, we simply multiply `Ti2C` by a 3-element tuple. The three integers `(nx, ny, nz)` repeat the unit cell in the x, y, and z directions, respectively.
```{code-cell}

sheet = Ti2C*(3,3,1) #

# Static image of the sheet
write('Ti2C_sheet.png', sheet, show_unit_cell=2, rotation=rotation)

# I've suppressed the interactive 3D view
# view(Ti2C, viewer='x3d', repeat=(4,4,1))

```

Additionally, the static image of the sheet is given below:

![](Ti2C_sheet.png)

## Building a 1D System - a Carbon Nanotube

A next example will be a carbon nanotube. ASE has functionality to build such structures in the `nanotube()` function.

```{code-cell}

from ase.build import nanotube
from ase.io import write # helps us save an image
from ase.visualize import view # 3D interactive image

atoms = nanotube(6, 0, length=4)

"""
   This is a easy way to make a simple (static) visualization.
"""
orientation='12y,-15z'
write('nanotube.png', atoms, show_unit_cell=2, rotation=orientation)
```

The static image of the nanotube is given below.

![](nanotube.png)

An interactive 3D image of the carbon nanotube is shown below.

```{code-cell}

# Interactive 3D visualization
view(atoms, viewer='x3d')

```

## Building Complex Crystals

The structures the `ase.build` module allows you to construct are fairly basic. For more advanced structures, we may follow the same strategies as for the complex molecules:
* Construct a crystal structure from a structure file (`*.cif`, `*.xyz`, etc.)
* Read the structure from simulation output
* Obtain a structure from a materials database

To read a structure from a file, use the `read()` function from the `ase.io` module. This module allows ASE to read from and write to files containing information about materials ([documentation](https://wiki.fysik.dtu.dk/ase/ase/io/io.html)).


## Exercises

:::{dropdown} Exercise 1: Form a nitrogen-vacancy center in a diamond crystal.

The nitrogen vacancy (NV) center in diamond is a point defect that can support a room-temperature qubit system. Write Python code to form a NV center in a diamond supercell, by doing the following:

> 1. Define a diamond primitive unit cell using `ase.build.bulk()`.
> 2. Use the primitive unit cell to form a supercell that is at least a three-by-three-by-three supercell.
> 3. Make a nitrogen substitution by swapping one C atom for a N atom.
> 4. Remove a C atom adjacent to the N substitution.
> 5. Provide a static and an interactive visualization of your crystal.

Below is one example your result could resemble. Here, I have performed a substitution on the atom at the origin, and I have removed an adjacent atom.

![](nv_center.png)

---

_Hints_:
1. You can substitute the $k$-th atom of an `ase.Atoms` object simply by reassigning its atomic symbol. For example, given an `Atoms` object `si_crystal` representing a pristine Si crystal, we can transform the $k$-th atom to a C atom using code like this:
```
si_crystal[k].symbol = 'C'
```
2. See the ([documentation](https://wiki.fysik.dtu.dk/ase/ase/atoms.html)) for the `ase.Atoms` class to learn how to delete atoms from an `Atoms` object.

A successfully-formed NV center would still require structural optimization.

:::


### Solutions

```{code-cell}
:tags: [hide-cell]

from ase.build import bulk

diamond = bulk('C', 'diamond')

crystal = diamond*(3,3,3)

crystal[0].symbol = 'N' # it's in-silico alchemy!
del crystal[1]

"""
   Static visualization.
"""
orientation='12y,-15z'
write('nv_center.png', crystal, show_unit_cell=2, rotation=orientation)

# Interactive 3D visualization
view(crystal, viewer='x3d')
```
