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

```{code-cell}

from ase.build import bulk
from ase.visualize import view

atoms = bulk('C','diamond', 3.57)

# This is an added comment
view(atoms, viewer='x3d')
```
