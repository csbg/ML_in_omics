# Instructions 1: ML basics in sklearn

## Setup
First load required packages
```python
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import anndata
import os
from scipy import stats
```

Set a working directory and a directory where we store the data. Replace `???`:
```python
os.chdir("???")
os.makedirs("data", exist_ok=True)
```
### Dataset download

Download a dataset. We will download a relatively small dataset from the CELL x GENE Website using `wget` (in `bash` and *NOT* in `python`!).
```bash
wget -O data/my_dataset_small.h5ad https://datasets.cellxgene.cziscience.com/9d09a542-672b-4cc0-b4d4-05d318f5705f.h5ad
```

### If you want to download a different dataset

Go to the CZ CELL x GENE Website `https://cellxgene.cziscience.com/`. Locate `Datasets`, find a dataset you are interested in, click on `Download`, and then copy/paste the URL, e.g.: `https://datasets.cellxgene.cziscience.com/9d09a542-672b-4cc0-b4d4-05d318f5705f.h5ad`. Look at the size on the website. Datasets that are very big will be difficult to analyze on your computer.

## Load and explore the data

Load the dataset using the `read_h5ad` function from the `anndata` package and store it in the variable `data`. Explore the anndata object (`X`, `obs`, and `var`). 
Note: You can write just the object name to get a summary or `print` the object.

Use the following code to summarize the different column of the `obs` to understand the different experimental variables.
```python
for col in data.obs.columns:
    print("---")
    print(col)
    vals = data.obs[col].unique()
    if isinstance(vals, pd.Categorical):
        print(vals.categories.to_list())
    else:
        print(vals[1:20])
```

Summarize the numbers of cell per cell type.
```python
for i in data.obs["cell_type"].unique():
    print(i + ":")
    print(sum(data.obs["cell_type"] == i))
```

## Normalize and filter data

Look at the data. What are the dimensions? Look at the first few expression values by extracting the first 10 rows and columns of the matrix and showing using the method `todense()`. What types of numbers are these?

Next, normalize the data. At each step, again look at the first few expression values to see how they changed.
1. Calculate the sum of reads per cell and devide the data by this sum: `data.X /= data.X.sum(axis=1)`. Note: If you have an error about CSR and CSC matrices, use `data.X = (data.X / data.X.sum(axis=1)).tocsr()`. This converts the sparse matrix format from COO to CSR, which may be required by anndata.
2. Multiply each value by 1 million (`10e6`).
3. Use `log1p` from `numpy` to log normalize the data.
4. Convert the sparse matrix to the Compressed Sparse Row format using the method `tocsr()`.

To filter the data, we will remove all genes that are not expressed in at least 10% of cells.
1. Use `data.shape` to get the number of cells.
2. Sum up the reads for each gene.
3. Test if the sum from #2 is greater than 10% of the number of cells from #1.
4. Filter the dataset to only those genes where this is the case.

## Train ML model

To setup your data and model, define:
- `X` as the normalized and filtered gene expression data (input data).
- `Y` as the `cell_type` from `data.obs`, after converting this colummn to a `list`: `data.obs["cell_type"].to_list()` (output).
- `model` using the function `LogisticRegression()`. Set the maximum iterations to 200. 

Next, fit the model to the data using the method `fit()` with the correct arguments. Note: you don't have to save the model at this step. The object will be updated.

## Predict and evaluate the model

To predict values for new data, use the method `predict()` of your model. 

Here, we will first predict on the data that was used for training. Store the prediction results in a new variable `Y_hat`.

Look at the continguency table for the predictions: `print(pd.crosstab(Y,Y_hat,margins = False))`.

Calculate the accuracy:
1. Using `==` to compare where `Y` is the same as `Y_hat`.
2. Use `sum()` to count the number of cases where this is `True` (will be counted as `1`).
3. Devide this sum by the total number of test cases.

## Train / test split

AI/ML algorithms should always be evaluated on unseed data. Split data and labels into train and test set using the function `train_test_split()`, keeping 30% of data for the test set.
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
```
Now define a new model and train it on the training data. Then report the continguency table and accuracy on the test data.

## Train ML model on a more difficult task

Distinguishing major cell types is relatively easy based on transcriptomics data. We will now address a more challenging problem. Instead of `cell_type`, now predict `author_cell_type`, which has a more fine-grained annotation of cells. Ensure to evaluate on the test set.

## More detailed evaluation

We will now evaluate in each class (cell type), how many predictions were correct.
1. Start an empty list: `eval = []`
2. Loop over `set(Y_test)`, i.e. unique labels (cell types) using the variable `ct`.
3. Get the indices of `ct` in `Y_test` correspond to this cell type using this list comprehension: `[i for i, cx in enumerate(Y_test) if cx == ct]`
4. Store the following in a `pandas` `DataFrame` called `res` that has one row:
    ```python
    res = pd.DataFrame({
        "cell_type": ct, 
        "accuracy": sum([Y_test_hat[i] == ct for i in idx]) / len(idx), 
        "number": len(idx)
    }, index=[0])
    ```
5. Append the `res` to `eval`.
6. Concatenate the DataFrames together using `eval = pd.concat(eval)`

Look at the result using: `print(eval)`. Now, plot the results, using the function `scatter()` from the `matplotlib.pyplot` package, with `eval.accuracy` on the x-axis and `eval.cell_type` on the y-axis.

Make a scatterplot of accuracy vs number of cells from `eval`. Next use `stats.pearsonr()` to correlate the two. 

Use `skm.classification_report()` to summarize different prediction performance metrics.

## ROC AUC evaluation

Now plot the ROC curve for each cell type and calculate the AUC.
1. Use the method `predict_proba()` from your model to get class probabilities for your test set.
2. Set up a large plot with 15 slots using `fig, axs = plt.subplots(5, 3, constrained_layout=True)`
3. Loop through the cell types using `for i,cl in enumerate(model.classes_):`. All next steps are within the loop:
    1. Store the false positive rate and true positive rate: `fpr, tpr, thresholds = skm.roc_curve([0 + (x == cl) for x in Y_test], Y_test_hat_probs[:,i])`
    2. Plot the rates against each other: `axs.flatten()[i].plot(fpr, tpr)`
    3. Add a plot title `axs.flatten()[i].set_title(cl, fontdict = {'color':'blue','size':7})`
    4. Calculate and print the AUC: `skm.auc(fpr, tpr))`
4. Outside of the loop, show the full plot: `plt.show()`

# EXERCISES
## Exercise 1.1
Describe:
1. the dimensions (method `shape()` or function `len()`)
2. types of data or object (function `type()`)
3. range of values
for the following objects:
- the anndata object `data` (including attributes `X`, `obs`, and `var`) after filtering
- the anndata object `data` before filtering genes (cells will remain unchanged)
- the input `X`
- the true output `Y`
- the training inputs `X_train`
- the test inputs `X_test`
- the training labels `Y_train`
- the test labels `Y_test`
- the predicted output `Y_hat` (only needed for one model)

## Exercise 1.2
Which other labels could you predict from this dataset?

## Exercise 1.3
1. Which task showed the best and worst performance?
2. Did you observe a difference when using a test set instead of evaluating on the training set?

## Exercise 1.4
1. Was there a difference in performance between the different classes?
2. Do accuracy, f1-score, and AUC agree on where performance was worst?
3. Does performance depend on class size (number of cells per cell type)?
