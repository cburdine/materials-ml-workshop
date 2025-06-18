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

# Application: XRD Crystal Structure Prediction

Now that we know how to create and train neural networks in PyTorch, let's get some practice applying them to solve an important problem in solid state crystallography: predicting crystal symmetries (in particular, the crystal space group) from powder x-ray diffraction patterns.

## X-Ray Diffraction (XRD)

X-ray crystallography is a powerful technique for characterizing the arangement of atoms in solid structure, first introduced by Paul Ewald and Max von Laue in 1912. It has since been applied to estimate the lattice constants of crystalling solids, and even to obtain the structure of complex organic molecules, such as proteins.

In X-ray crystallography, an X-ray beam is focused into a crystalline material, causing the beam to diffract from the periodically arranged nucleii at specific angles $\theta$. These diffraction angles $\theta$ are given by Bragg's diffraction law

$$n \lambda = 2 d\sin(\theta)$$

where $\lambda$ is the wavelength of the incident beam, $n$ is a small integer (the diffraction order), and $d$ is the distance between two planes of atoms in the material.

![bragg_diffraction_law](bragg_diffraction_law.png)

For each material, there are many different values of $n$ and $d$ that satisfy Bragg's law, forming diffraction "peaks" at several different angles $\theta$. The angle and intensity of these peaks depend on the symmetries and lattice constants of a material; however reverse-engineering the full 3D structure of a crystal from these peaks is known to be a complex problem.

In this application, we will attempt to use simple neural networks to predict the space group of known materials from their XRD spectrum. The space group of a material describes all of the symmetries of the material's crystal lattice. For three dimensional materials, there are up to 219 unique space groups our neural network will need to distinguish based on XRD data alone.


You can download the dataset for this section using the following Python code:

```
import requests

CSV_URL = 'https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/neural_networks/xrd_dataset_cleaned.csv'

r = requests.get(CSV_URL)
with open('matml_supercon_cleaned.csv', 'w', encoding='utf-8') as f:
    f.write(r.text)
```

Alternatively, you can download the CSV file directly [here](https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/neural_networks/xrd_dataset_cleaned.csv).

## Loading the Dataset

```{code-cell}
:tags: [hide-input]
import pandas as pd

# load dataset into a pandas DataFrame:
XRD_CSV = 'xrd_dataset_cleaned.csv'
data_df = pd.read_csv(XRD_CSV)

# show dataframe in notebook:
display(data_df)
```

```{code-cell}
import ast # <- parses Python literals from text

# Generate a list of elements in the dataset:
ELEMENTS = set()
for v in xrd_df['composition'].values:
    ELEMENTS |= set(ast.literal_eval(v).keys())
ELEMENTS = sorted(ELEMENTS)

# Generate a list of the crystal systems in the dataset:
CRYSTAL_SYSTEMS = sorted(set(xrd_df['crystal_system']))

# Generate a list of the symmetry symbols:
SYMM_SYMBOLS = sorted(set(xrd_df['symmetry_symbol']))

# print the sizes of ELEMENTS and CRYSTAL_SYSTEMS
print('Number of elements:', len(ELEMENTS))
print('Number of symmetry symbols:', len(ELEMENTS))
print('Number of crystal systems:', len(CRYSTAL_SYSTEMS))
```

```{code-cell}
:tags: [hide-output]
print(SYMM_SYMBOLS)
```

```{code-cell}
# Generate a map from each symmetry symbol to
# the corresponding crystal system
SYMM_SYMBOL_MAP = {}
for _, row in xrd_df.iterrows():
    system = row['crystal_system']
    symm = row['symmetry_symbol']
    SYMM_SYMBOL_MAP[symm] = system
```

```{code-cell}
import matplotlib.pyplot as plt

def parse_xrd_data(row):
    peaks = ast.literal_eval(row['xray_peaks'])
    ints = ast.literal_eval(row['xray_intensities'])
    return peaks, ints
    
def plot_xrd(peaks, intensities):
    
    plt.figure()
    for x, y in zip(peaks, intensities):
        plt.plot([x,x], [0, y], color='b', linewidth=2)
    
    plt.axhline(color='b')
    plt.xlim((0,90))
    plt.grid()
    plt.ylabel('Intensity [arb. units]')
    plt.xlabel(r'Diffraction Angle $2\theta$ [degrees]')
    plt.show()
```

```{code-cell}
example_idx = 1234

plot_xrd(*parse_xrd_data(xrd_df.iloc[example_idx]))
```

```{code-cell}
INTENSITY_SCALE = max([
    max(parse_xrd_data(row)[1])
    for _, row in xrd_df.iterrows()
])

print(INTENSITY_SCALE)
```

```{code-cell}
import numpy as np

def vectorize_composition(composition, elements):
    """ converts an elemental composition dict to a vector. """
    total_n = sum(composition.values())
    vec = np.zeros(len(elements))
    for elem, n in composition.items():
        if elem in elements:
            vec[elements.index(elem)] = n/total_n
    return vec

def vectorize_crystal_system(crystal_system, systems):
    """ converts a crystal system to a vector. """
    vec = np.zeros(len(systems))
    if crystal_system in systems:
        vec[systems.index(crystal_system)] = 1.0
        
    return vec

def vectorize_symmetry(symmetry_symbol, symbols):
    """ converts a symmetry symbol to a vector. """
    vec = np.zeros(len(symbols))
    if symmetry_symbol in symbols:
        vec[symbols.index(symmetry_symbol)] = 1.0

    return vec
```

```{code-cell}
def vectorize_xrd_spectrum(peaks, intensities, intensity_scale=100, bins=90):
    hist, _ = np.histogram(
        peaks, bins=bins,
        range=(0, 90),
        weights=intensities)
    
    return hist / intensity_scale
```

```{code-cell}
def parse_data_vector(row):
    """ parses x and y vectors from a dataframe row """
    
    # parse the xray peaks and intensities
    peaks, ints = parse_xrd_data(row)
    
    # parse feature vector (x):
    x_vector = vectorize_xrd_spectrum(peaks, ints)
    
    # parse label vector (y):
    y_vector = vectorize_symmetry(row['symmetry_symbol'], symbols=SYMM_SYMBOLS)
    #y_vector = vectorize_crystal_system(row['crystal_system'], systems=CRYSTAL_SYSTEMS)
    
    return x_vector, y_vector
```

```{code-cell}
import torch
import torch.nn as nn

# Define the neural network class (XRDNet)
class XRDNet(nn.Module):

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """ Constructs a feed-forward neural network with many hidden layers"""
        super().__init__()

        layer_sizes = [input_size]
        layer_sizes.extend(hidden_layer_sizes)
        layer_sizes.append(output_size)
        
        self.hidden_layers = nn.ParameterList([
            nn.Linear(size_in, size_out)
            for size_in, size_out in 
            zip(layer_sizes[:-1], layer_sizes[1:])
        ])

        # define hidden layer activation function:
        self.activation = nn.SiLU()

        self.out_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        """ Estimates the relative log-likelihood (logit) of each output feature"""
        for layer in self.hidden_layers[:-1]:
            x = self.activation(layer(x))
        
        out = self.hidden_layers[-1](x)

        return out
    
    def classify(self, x):
        """ Estimates the normalized output probability of each output feature"""
        
        out = self.output_logits(x)
        return self.out_activation(out)
```

```{code-cell}
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def eval_model_loss(model, data_loader, loss_fn):
    
    loss = []
    with torch.no_grad():
        for z_batch, y_batch in data_loader:
            loss.append(loss_fn(model(z_batch), y_batch))

    mean_loss = torch.stack(loss).mean().item()

    return mean_loss

def eval_model_accuracy(model, data_loader):
    acc = []
    with torch.no_grad():
        for z_batch, y_batch in data_loader:
            pred_class = model(z_batch).argmax(dim=-1)
            true_class = y_batch.argmax(dim=-1)
            acc.append((pred_class == true_class).to(torch.float32).mean())

    mean_acc = torch.stack(acc).mean().item()

    return mean_acc
```


```{code-cell}
from torch.utils.data import random_split

dataset = XRDDataset(xrd_df)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
```

```{code-cell}
model = XRDNet(
    input_size=len(dataset[0][0]),
    hidden_layer_sizes= [512, 512, 256, 128],
    output_size=len(dataset[0][1]),
)
```

```{code-cell}
def fit_model(model, train_dataset, val_dataset, n_epochs=100, batch_size=64, lr=1e-3, wd=1e-3):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    
    # create a dict to record losses during training
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # main training loop (fixed number of epochs)
    for epoch in range(n_epochs):
        
        # apply stochastic gradient descent step to each batch in dataset
        print(f'Epoch {epoch}')
        epoch_losses = []
        for z_batch, y_batch in tqdm(train_loader):
            
            # zero optimizer gradients
            optimizer.zero_grad()
    
            # generate batch prediction
            y_hat_batch = model(z_batch)
            
            # compute loss
            loss = loss_fn(y_hat_batch, y_batch)
            epoch_losses.append(loss.item())
    
            # backpropagate loss
            loss.backward()
            optimizer.step()

        # evaluate epoch training and validation losses:
        train_loss = eval_model_loss(model, train_loader, loss_fn)
        val_loss = eval_model_loss(model, val_loader, loss_fn)
        print('Acc: ', eval_model_accuracy(model, val_loader))
        print(f'Train loss: {train_loss}; Val loss: {val_loss}')

        # record losses in history dictionary
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
    
    return history
```

```
history = fit_model(
    model, 
    train_dataset, 
    val_dataset, 
    batch_size=512,
    n_epochs=50,
    lr=1e-3,
    wd=1e-1
)
```


```{code-cell}
train_acc = eval_model_accuracy(model, train_dataset)
val_acc = eval_model_accuracy(model, val_dataset)
test_acc = eval_model_accuracy(model, test_dataset)

print(train_acc)
print(val_acc)
print(test_acc)
```