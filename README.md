# Linear SVM on the UCI Heart Disease Dataset

This project implements a **linear soft-margin Support Vector Machine (SVM)** from scratch for binary classification on the **UCI Heart Disease dataset**.

The implementation follows the homework requirements:
- normalize the training and test features using **training-set statistics only**
- train a **primal soft-margin SVM**
- solve the optimization problem using **CVXPY**
- evaluate accuracy on both train and test sets
- analyze the effect of different values of **C**

## Problem

We are given the Heart Disease dataset and asked to implement a **linear SVM without using scikit-learn**.

The required tasks are:
1. Normalize the dataset using the **mean and standard deviation from the training set**
2. Implement `trainSVM(X_train_normalized, y_train, C)` to solve the primal soft-margin SVM:
   \[
   \min_{w,b,\xi} \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^{N} \xi_i
   \]
   subject to:
   \[
   y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
   \]
3. Implement `evalSVM(X_eval, y_eval, w, b)` to compute classification accuracy
4. Train and evaluate the model for:
   \[
   C = a \times 10^q, \quad a \in \{1,3,6\}, \; q \in \{-4,-3,-2,-1,0,1\}
   \]
5. Plot:
   - Training Accuracy vs C
   - Test Accuracy vs C

## Dataset

- Source: UCI Heart Disease dataset
- Samples: 303
- Input features before preprocessing: 13
- After one-hot encoding: 22 features
- Train/Test split after notebook preprocessing:
  - `X_train`: `(216, 22)`
  - `X_test`: `(83, 22)`

## Key Results

### Normalization statistics from training data
- First feature mean: **54.99074**
- First feature standard deviation: **9.07784**
- Last feature mean: **0.50462**
- Last feature standard deviation: **0.49997**

### Best accuracies
- **Maximum training accuracy:** `0.8889` at `C = 10`
- **Maximum test accuracy:** `0.8554` at `C = 0.003`

### SVM parameters for selected C values

#### For `C = 1`
- First three weights: `[-0.01280, 0.51706, 0.27813]`
- Bias: `[0.08109]`
- First three slack variables: approximately `[0, 0, 0]`

#### For `C = 0`
- First three weights: `[3.09523e-06, -8.18802e-06, -9.46615e-06]`
- Bias: `[-10.44762]`
- First three slack variables: `[429.58840, 434.02071, 414.34670]`

## Why normalize using only the training set?

The mean and standard deviation must be estimated from the **training data only** and then applied to both training and test sets. This avoids leaking information from the test set into the model-building pipeline and keeps the test data as a truly unseen distribution during evaluation.


## Code to reproduce the results

### 1. Clone the repository
```bash
git clone https://github.com/your-username/linear-svm-heart-disease-from-scratch.git
cd linear-svm-heart-disease-from-scratch
