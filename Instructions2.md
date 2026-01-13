# Instructions 2: weight optimization in pytorch

## Setup
First load required packages
```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import anndata
import os
```

Set a working directory.
```python
os.chdir("TODO")
```

## Load and prepare data

First, load the data and then subset them to Astrocytes and Glioblasts in order to limit the dataset to 2 classes.

```python
data = data[data.obs.author_cell_type.apply(lambda x: x in ["Astrocytes", "Glioblasts"])]
```

Next, filter (remove genes with few reads) and normalize the data (logCPM) the same way done on day 1.

## Set up pytorch tensors

Convert the input and output to pytorch tensors.

For the input, we need a dense numpy matrix.
```python
X = torch.Tensor(data.X.todense())
```

For the output, we need a dummy encoding (one hot encoding) of the label.
```python
Y = torch.Tensor(np.array([[0 + (x == "Astrocytes") for x in data.obs.author_cell_type]])).transpose(0,1)
```

## Scale data

We will scale the features (genes) to equal mean (0) and variance (standard deviation of 1). This is done by subtracting the mean and dividing by the SD.
```python
def scale(x):
    m = x.mean(axis=0, keepdim=True)
    s = x.std(axis=0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return(x)

X = scale(X)
```

## Define ML model

Below is a bare-bones pytorch model.
- The `__init__()` methods initializes the model.
- The `forward()` method does the forward pass.
- The backward pass will be handled automatically by pytorch.
- The model only has one linear layer. It will once multiple the input matrix (`X`) by weights (`W`), and sum these values to produce output (`Y`).

Your tasks:
1. Replace the `???`
2. Add as sigmoid transformation (`nn.Sigmoid()`) that will be executed after the linear transformation. Note: Add this to both methods (`__init__` and `forward`) similar to the `linear1` layer..

```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(???, 1)
        
    def forward(self, x):
        x = self.linear1(x)
        return x
```

## Model and learning setup

Initialize the model
```python
model = SimpleModel(input_dim=???)
```

Define loss calculation function (`criterion`).
```python
criterion = nn.BCELoss()
```

Define the optimizer (that performs the gradient descent). The arguments are (1) the model parameters (weights) to optimize and (2) the learning rate (lr).
```python
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

Look at the weights of the linear layer, which are in the `weight` attribute of the `nn.Linear()` object.

## Assess performance

Apply the model the data (one forward pass). Look at the values of outputs.
```python
outputs = model(X)
```

Calculate loss compared to true labels. Replace `???`
```python
loss = criterion(???, ???)
```

Look at the loss using: `loss.item()`. Also calculate at the accuracy. 
1. Calculate output class probabilities
2. Convert to a boolean by testing whether the value is greater than 0.5
3. Add `0` to this tensor to convert boolean to numeric (1 if True; 0 if False)
4. Test if the value is equal to the true labels.
5. Use the method `sum()` to sum these boolean results (again True will be 1 and False will be 0).
6. Devide by the number of samples.

*NOTE:* In this lesson, for simplicity, we are not splitting train / test set. In real-life scenarios, you should split them!


## Train the model

Some setup for the training. After each line, look at the weights and see if they changed. After the training step, look at the accuracy again (using the same code as above).
```python
model.train() # tell pytorch that we will train
optimizer.zero_grad() # remove previous gradients (if they were already calculated)
loss.backward() # Do one backwards pass to calculate gradients
optimizer.step() # Optimize weights
```

## Many training steps

Now put all training steps into a loop (10 iterations aka "training epochs"). At each iteration you have to:
1. calculate output
2. calculate loss
3. do one backward pass
4. optimize
5. print loss

Afterwards, look at the weights and accuracy.

## A more complex model

Modify the simple model above to include:
1. Layer 1:
    1. Linear layer 1 (from input nodes to 50 hidden nodes)
    2. Sigmoid layer 1 (output of layer 1 and input of layer 2)
2. Layer 2:
    1. Linear layer 2 (from 50 hidden nodes to 1 output node)
    2. Sigmoid layer 2 (output)

Train the model as done above and assess accuracy after the same number of epochs. 

## List of layers (bonus)

A more elegant version to code deep networks is to add layers to a layer list, which is then used in a loop in the forward pass:
```python
class OurNN_ListVersion(nn.Module):
    def __init__(self, input_dim):
        super(OurNN_ListVersion, self).__init__()
        layer_list = []
        layer_list.append(nn.Linear(???,???))
        layer_list.append(???)
        layer_list.append(???)
        layer_list.append(???)
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

Modify the code above and train the model as done above and assess accuracy after the same number of epochs.


# EXERCISES
## Exercise 2.1
Describe:
1. the dimensions (method `shape()` or function `len()`)
2. types of data or object (function `type()`)
3. range of values
for the following objects:
- the anndata object `data` after filtering genes and cells
- the input `X`
- the true output `Y`
- the predicted output `outputs`
- the weights within the `model` object

## Exercise 2.2
At which step were the weights of the model changed in the simple model?

## Exercise 2.3
Was the learning process successful? How did the loss and the accuracy change?
