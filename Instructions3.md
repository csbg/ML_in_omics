# Instructions 3: multi-class predictions in pytorch

## Setup
Load packages and prepare data as in lesson 2. Do not remove cells, we will here analyze all cells and cell types.

## Setup data and tensors

Prepare the input data `X` the same way as in lesson 2 (normalize, filter, scale).

For the labels, we will create a dummy encoding of multiple classes. This will produce a matrix (tensor) that of `cells` x `cell types` of `[0;1]`, where each cell has a 1 in only one cell type and 0 everywhere else. To do so, follow these steps:
1. Use the function `get_dummies()` from `pandas` on the labels, creating a data frame of boolean statements.
2. Add `0` to this data frame to convert this to `[0;1]` encoding.
3. Convert this data frame to numpy using the method `to_numpy()`.
4. Convert the numpy matrix to a tensor `Y`.

## Define ML model

Now set up a neural network as in lesson 2. We will remove the the final sigmoid layer and end with a final linear layer. To obtain class probabilities, we will calculate softmax, but this is done by the loss function (see below) and does not have to be done in the model.

## Model training

Train the model as seen in lesson 2 with the following modifications:
1. Use `nn.CrossEntropyLoss()` instead of `nn.BCEWithLogitsLoss()`. This will do the softmax and calculate the loss.
2. Use `lr=0.1` in the optimizer.
3. When training over many epochs, store your loss in a list. Initialize the list before starting `loss_list = []` and then add to it in the training loop using `loss_list.append(???)`.
4. Train for 50 epochs.
5. Plot the loss using a Scatterplot. Show the loss on the y-axis and the epoch on the x-axis.

## Modifying the learning rate.

Now retrain the model from the start, but reduce the learning rate to `lr=0.01`, but store the lost in a list with a different name. Make a plot of both lists (you just need to call scatterplot twice). Compare the curves.

# EXERCISES
## Exercise 3.1
Describe:
1. the dimensions (method `shape()` or function `len()`)
2. types of data or object (function `type()`)
3. range of values
for the following objects:
- the anndata object `data` after filtering genes
- the input `X`
- the true output `Y`
- the predicted output `outputs`
- the weights within the `model` object

## Exercise 3.2
Double check that your `Y` object really just has one `1` per cell and otherwise `0`. Make a sum of values per cell. 

## Exercise 3.3
Was the learning process smooth and as expected? How did the learning rate influence the process?