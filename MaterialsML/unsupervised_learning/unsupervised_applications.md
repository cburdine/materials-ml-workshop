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

# Application: Classifying Superconductors

You can download the dataset for this section using the following Python code:

```{code-cell}
import requests

CSV_URL = 'https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/unsupervised_learning/matml_supercon_cleaned.csv'

r = requests.get(CSV_URL)
with open('bandgaps.csv', 'w', encoding='utf-8') as f:
    f.write(r.text)
```

Alternatively, you can download the CSV file directly [here](https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/unsupervised_learning/matml_supercon_cleaned.csv).
