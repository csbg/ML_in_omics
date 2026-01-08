# Instructions 5: Random forest and interpretability

## Setup
Load packages and prepare data as in lesson 1 with the following modification: 
1. Filter lowly expressed genes using a cutoff 0.5: `data = data[:,data.X.sum(axis=0) > data.shape[0] * 0.5]`
2. Filter the dataset to astrocytes and glioblasts as in lesson 2.

## Setup data and tensors
Prepare `X` and `Y` the same way as in lesson 1.

## Setup data and tensors
Prepare `X` and `Y` the same way as in lesson 1.

## Models / hyperparameters
Define a random forest model using `RandomForestClassifier` from sklearn with 10 estimators. We will not optimze any hyperparameters here.

## Training / testing
Use 25% of the data for testing. Train the model and test the accuracy. 

## Feature importance
Obtain the permutation feature importance of your model.
1. Use the function `permutation_importance` from sklearn on your model.
2. Run this on your test data. Note: You have to convert the data to a dense numpy array using: `np.asarray(x.todense())`.
3. Use 5 repeats.
4. Set a random_state.
5. Store the results in the variable `feature_importance`. Explore this variable.

## Explore feature importance values
Plot a histogram `plt.hist` of the mean feature importances (as calculated by permutation_importance).

## Data of most important feature
Now we will plot the expression data of the most important feature:
1. Take the absolute value of the mean feature importances.
2. Run the function `np.argsort` on `axis=0` to identify the most important features and save this into `idx`. This will provide an array with the same dimensions but the values are the indices of the sorted values, i.e. the index of the smallest importance values is on position 0 and the biggest one is on the last position.
3. Get the indices of the top 10 features (last 10 values from the above object) into the variable `i`.
4. Look at the mean feature importances of these features.
5. Use `data.var.iloc[i]` to look at the gene information of these features.
6. From the expression data, get the expression values of the top gene (last gene of `i`) into the variable `data_f1`.
7. Look at the `shape` of `data_f1`.
8. Convert this to a simpler shape using `np.asarray(data_f1.todense())[:,0]`. This (1) converts the sparse matrix to a dense matrix, (2) creates a numpy array out of the matrix, and (3) converts this from a 2-D to a 1-D array.
9. Separate expression data by astrocytes and glioblasts:
    - From `data.obs`, create a boolean vector (True / False) for astrocytes and use this to extract only the astrocyte-expression data from `data_f1`.
    - Repeat the same for glioblasts.
    - Combine both arrays in a list.
11. Create a boxplot from the list you just created.

# EXERCISES

## Exercise 1.1
Describe:
1. the dimensions (method `shape()` or function `len()`)
2. types of data or object (function `type()`)
3. range of values
for the following objects:
- the training inputs `X_train`
- the test inputs `X_test`
- the training labels `Y_train`
- the test labels `Y_test`
- all objects within the dict returned by `feature_importance`
- the expression data for our top gene `data_f1`

## Exercise 1.2
How many genes have a high importance?

## Exercise 1.3
Did the most important genes make sense biologically?

## Exercise 1.4
Did the expression data of the top gene explain why this was picked by the algorithm?