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

# Pymatgen and the Materials Project API

## Pymatgen - Python Materials Genomics

![](pymatgen.svg)

[Pymatgen](https://pymatgen.org) is a Python package similar to `ase`. `pymatgen` is designed to support VASP and ABINIT. Like `ase`, we will not be able to cover much of the extensive functionality of `pymatgen`.

:::{Note}

Our use of `pymatgen` will be very limited. Nonetheless, we will install `pymatgen` for use as a converter:  we can obtain crystal structure data from the Materials Project and convert it to an `ase` `Atoms` object.
:::

### Installation

If you need to, please install `pymatgen`, please do so using a command such as `pip3 install pymatgen`.


## MPRester - The Materials Project API

![](materialsproj.png)

[The Materials Project API](https://materialsproject.org/api) allows a user to query information from [the Materials Project](https://materialsproject.org).

:::{admonition} What is an API?
:class: note, dropdown

"API" stands for *application programming interface*. An API is a set of commands defined to allow programmatic access to a server that archives or generates data for users.

APIs are especially useful for automating queries. They make getting large amounts of information much more efficient than point-and-click manual access to a web server via an Internet browser.
:::

### Installation

The Materials Project API is coded in the Python package `MPRester`. To use ASE, you must first install the `ase` Python module. You may use a command such as `pip3 install mp-api` to do this.

::::{important}
:::{note}
An API key is required to use `MPRester`.
* You must be logged in on [materialsproject.org](https://materialsproject.org) to obtain your API key.
* You can obtain your personal API key from your Materials Project [Dashboard](https://next-gen.materialsproject.org/dashboard), or you can get it from the [documentation page](https://materialsproject.org/api#documentation).
* Your API key is a long alphanumeric string (about 30 characters) that you must use every time you wish to query the Materials Project programmatically via the API.
:::
::::

### Using `MPRester`

To use your API key, it is helpful to store its value in a string, like this:
```{code-cell}
# Save your Materials Project API as a string
MP_API_KEY = '---your api key here ----'
```

```{code-cell}
:tags: ["remove-cell"]

import os

MP_API_KEY = os.environ['MP_API_KEY']
```

Once the API key is stored as a string in a variable (here, we used the variable `MP_API_KEY`), that variable is used as a parameter with `MPRester` to obtain information from the Materials Project. The syntax is to define a code block using code like `with MPRester(MP_API_KEY) as mpr:`. Following this line, indented lines define a code block in which we can access the API methods using the syntax `mpr.some_method()`. Let's start with some examples of `MPRester` usage.

#### Example: Get the Crystal Structure of a Specific Material

In the Materials Project, each material has a unique identifier, known as its Materials Project ID (MPID). When you want information about a single material, it is reasonable to perform a manual search using the [Materials Explorer](https://materialsproject.org/materials) for the material so you can find its MPID. Then, you can use the MPID along with `MPRester` to automate queries about the material.

As an example, YBa2Cu3O7 has `mp-20674` as its MPID. We can obtain its crystal structure directly using this MPID. To do this, we do the following:

```{code-cell}
:tags: ["hide-output"]

from mp_api.client import MPRester
import os
import pymatgen as pmg

from ase.io import write

MPID = 'mp-20674' # Materials Project ID number

"""
  The 'with ...' statement defines an MPRester code block. Subsequent
  indented statements belong to the code block, and the object mpr
  may be used within the code block.
"""
with MPRester(MP_API_KEY) as mpr:
    # Get only the structure for YBa2Cu3O7
    structure = mpr.get_structure_by_material_id(MPID)

```

The `MPRester` method, `get_structure_by_material_id()`, accessable as `mpr.get_structure_by_material()`, queries the Materials Project and returns the crystal structure, storing it in an object `structure`. Having accessed the Materials Project, we no longer require the `mpr` object. We can now exit the `MPRester` code block by resetting the indentation.

Next, we provide code to inspect the structure we downloaded. What is its data type? How can we use it?

```{code-cell}

"""
  Reset the indentation (exith the 'with' block), and examine the
  structure we obtained.
"""
print('\nWhat is the data type of the structure we obtained?')
print(type(structure))

print('\nWhat is the structure we obtained?')
print(structure)

```

The output of the above code cell indicates that the data in `structure` is in a format compatible with `pymatgen`. `pymatgen` has a tool to convert the `pymatgen` structure to an `ase` object, which we can visualize.

```{code-cell}

from pymatgen.io.ase import AseAtomsAdaptor as aaa

"""
   We use pymatgen to convert the structure to an
   ASE object. Since our work with the MaterialsProject
   API is complete, this can be done outside the WITH
   block.
"""
crystal = aaa.get_atoms(structure) # convert pymatgen to ase

# Make a static visualization
orientation='90x,75y,-9x'
write('YBa2Cu3O7_structure.png', crystal, show_unit_cell=2,
      rotation=orientation)

```

![](YBa2Cu3O7_structure.png)

The following code creates an interactive visualization for the downloaded crystal structure.

```{code-cell}
from ase.visualize import view

# Interactive 3D visualization
view(crystal, viewer='x3d')

```

Having obtained the crystal structure, we can now use it in a variety of ways:
* Use it within an atomistic simulation
* Use `ase.io.write()` to save the structure in a structure file (`*.cif`, `*.xyz`, etc.)

#### Example: Getting the Band Structure for a Material

We can use the `MPRester` class to obtain band structures for a material:
```{code-cell}
:tags: ["hide-output"]

mpid = "mp-149" # this is the MPID for silicon crystal (diamond lattice)

with MPRester(MP_API_KEY) as mpr:
    bs = mpr.get_bandstructure_by_material_id("mp-149")
```

This returns a `pymatgen` band structure object, and `pymatgen` has a tool for plotting this structure.
```{code-cell}
from pymatgen.electronic_structure.plotter import BSPlotter

# plot & show the band structure we obtained
BSPlotter(bs).get_plot().show()

```


#### Example: Searching using `MPRester`

Materials Project data can be queried in two ways:
* through a specific (list of) MPID(s), and/or
* through property filters (e.g. band gap less than 0.5 eV)

When querying a list of MPIDs, we use the following syntax:
```{code-cell}
:tags: ["hide-output"]

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.summary.search(material_ids=["mp-149", "mp-13", "mp-22526"])

```

Here, each material entry in the Materials Project has summary data, and we are simply searching the summary data using `mpr.summary.search()`. Since we queried for a list of MPIDs, we store in `docs` a list of "documents" (formally, a list of `MPDataDoc` objects).

We can now reference an individual document and extract its properties. We'll use a `for` loop to list the MPID and chemical formula for each search hit:

```{code-cell}

print('Our query returned {0} docs.'.format( len(docs) ))

for idx, mat_doc in enumerate(docs):
    print('Item {0}: MPID = {1} (formula: {2})'.format(idx,
                                                       mat_doc.material_id,
                                                       mat_doc.formula_pretty))

```

What properties (`'material_id'`, `'formula_pretty'`, etc.) are available for search in the summary data? We can obtain a list of document properties using the following syntax:
```{code-cell}

print(mpr.summary.available_fields)
```

Next, we query using property filters. We apply the following filters:
* Materials containing Si and O
* Materials with a band gap no greater than 1.0 eV but no less than 0.5 eV
* Instead of all available summary fields, we'll only ask for a few: `"material_id"`, `"formula_pretty"`, `"band_gap"`.
```{code-cell}
:tags: ["hide-output"]

with MPRester(MP_API_KEY) as mpr:
    docs = mpr.summary.search(elements=["Si", "O"],
                              band_gap=(0.5, 0.75),
                              fields=["material_id", "formula_pretty",
                                      "band_gap"])

example_doc = docs[0]
# initial_structures = example_doc.initial_structures

```

To see what our search turned up, we can use some simple code, like this. We first find out how many hits our query returned using `len(docs)`, and then we print only the first `N` hits, where we set `N = 10`.

```{code-cell}

N = 10
print('Our query returned {0} docs.'.format( len(docs) ))
print(f'Printing only the first {N} results:')

for idx in range(0,N):
    mat_doc = docs[idx]
    print('Item {0}: MPID = {1} ({2}), band gap = {3:6.4f} eV'.format(idx,
                                             mat_doc.material_id,
                                             mat_doc.formula_pretty,
                                             mat_doc.band_gap))

```


## Exercises

:::{dropdown} Exercise 1: Obtain the electronic density of states (DOS) for YBa2Cu3O7. Plot it using `pymatgen`.

---

_Hints_:
1. To find the code to obtain the DOS, a Google search such as "MPRester DOS example" may help, or perhaps you can try asking a chatbot.
2. Once you have obtained a DOS and saved it as, say, `some_DOS`, you can plot it using code such as this:
```
from pymatgen.electronic_structure.plotter import DosPlotter
import matplotlib.pyplot as plt

with MPRester(MP_API_KEY) as mpr:
    some_DOS = <your code to get the DOS>

# obtain a DosPlotter object
Plotter =  DosPlotter()

# add the DOS to the plotter
Plotter.add_dos('DOS', some_DOS)

"""
   Choose appropriate numbers for:

       E_lo and E_hi, the upper and lower limits
           of the domain for your DOS plot.

       MaxDensity, the upper limit for the range of
           your DOS plot.

   This may require some trial and error!
"""
Plotter.get_plot(xlim=(E_lo, E_hi), ylim=(0, MaxDensity))
plt.show()
```

:::


### Solutions

#### Exercise 1: YBCO DOS

```{code-cell}
:tags: [hide-cell]

from mp_api.client import MPRester # client for Materials Project
from pymatgen.electronic_structure.plotter import DosPlotter
import matplotlib.pyplot as plt

YBCO = 'mp-20674' # Materials Project ID number

with MPRester(MP_API_KEY) as mpr:
    YBCO_DOS = mpr.get_dos_by_material_id(YBCO)

print(YBCO_DOS)

# plot & show DOS we obtained
Plotter =  DosPlotter()

Plotter.add_dos('DOS', YBCO_DOS)

Plotter.get_plot(xlim=(-10, 10), ylim=(0, 30))
plt.show()
```

<!---
#### Exercise 2: Find TMDs

```{code-cell}
:tags: [hide-cell]

# Define the chemical elements for transition metals and chalcogens
transition_metals = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo",
                     "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
                     "Hg", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv",
                     "Ts", "Og"]

chalcogens = ["S", "Se", "Te"]


with MPRester(MP_API_KEY) as mpr:
      M = 'W' # transition_metals[0]
      X = 'Se' # chalcogens[0]

      docs = mpr.summary.search(chemsys=f"{M}-{X}",
                                nelements = 2,
                                crystal_system = 'Hexagonal', # symbol = 'P6_3/mmc',
                                fields=["material_id", "band_gap", "symmetry",
                                        "formula_pretty"]
                                )

ndocs = len(docs)

print(f"Found {ndocs} materials.")

for doc in docs:
    print("{0} ({1}), symmetry: {2}".format( doc.material_id,
                                             doc.formula_pretty,
                                             doc.symmetry))
```
-->
