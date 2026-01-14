# Instructions 4: Hyperparameter optimization

## Setup
Load packages and prepare data as in lesson 1.

## Setup data and tensors
Prepare `X` and `Y` the same way as in lesson 1.

## Models / hyperparameters
Define two models `model1` and `model2`. Both should be logistic regression models trained the same way as in lesson 1 with the following modifications:
1. `model1` uses an l2 penalty and `C=1000.0` (little regularization)
2. `model2` uses an l2 penalty and `C=1` (more regularization)
So, here we will optimize one hyperparameter, which is the regularization strength (`C`), evaluating only 2 values / settings. 

## Cross validation (CV)
Do 5-fold cross validation.
1. Split the data (`X` and `Y`) into test and CV data using `train_test_split` as in lesson 1, keeping 25% for testing.
2. Define a variable `model_choice` as 0. This will help us decide between the two models.
3. Run a loop with 5 iterations. Within the loop:
    1. Split the CV data into validation and train data using `train_test_split`, keeping 30% for validation (~25% of the original data).
    2. Fit both models on the training data.
    3. Predict class on the validation data.
    4. Calculate accuracy on validation data for both models (Note: Do NOT use test data here!).
    5. Print the accuracy of both models.
    6. If accuracy of model 1 is bigger than accuracy of model 2, then add 1 to `model_choice`. Oherwise, subtract 1.
4. Once the loop has finished, you know which model did better by looking at `model_choice`.

## Evaluate
Calculate the accuracy of the better model on the test data. This is your final prediction accuracy on unseen data.

## Check effect of regularization
Let's look at the effect of the l2 regularization on the weights.
1. Train each model on all data used in the cross-validation (train and validation data).
2. For each model, get the attribute `coef_` (weights) and make a histogram of the values for one class.
3. Take the absolute value of the coefficients (`np.abs()`), and calculate the mean (method: `mean()`) for each class. Then make a scatterplot, comparing these averages between the two models.


# EXERCISES
## Exercise 1.1
Describe:
1. the dimensions (method `shape()` or function `len()`)
2. types of data or object (function `type()`)
3. range of values
for the following objects:
- the training inputs `X_cv`
- the test inputs `X_test`
- the training labels `Y_cv`
- the test labels `Y_test`
- the model coefficients of model 1

## Exercise 1.2
Which model performed better? In how many folds was it better?

## Exercise 1.3
Did you see a difference in accuracy during hyperparameter optimization compared to testing?

## Exercise 1.4
What did the regularization do to the coefficients? How did it change the values?
