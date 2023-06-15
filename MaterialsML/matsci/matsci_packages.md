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
2. Python Materials Genomics (pymatgen)
3. The Materials Project API (MPRester)

In the following pages, we will focus on ASE and MPRester. ASE will be used to create/visuzalize simple atomic structures. MPRester will be used to obtain data from the Materials Project database, and pymatgen will be used to convert data obtained using MPRester into ASE structures, or to plot data obtained using MPRester.

Both ASE and pymatgen may be used to run atomistic calculations (density functional theory, molecular dynamics, etc.), but this is beyond the scope of this workshop.
