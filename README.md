# Mushroom Classification Web App

A simple Streamlit app for binary classification: Predict if mushrooms are edible or poisonous using machine learning models.

## Description

This app uses the Mushrooms (fake) dataset to train and evaluate classifiers like SVM, Logistic Regression, and Random Forest. Users can tweak hyperparameters via the sidebar and visualize metrics (accuracy, precision, recall) along with plots like confusion matrices and ROC curves.

Key features:
- Interactive hyperparameter tuning
- Model comparison
- Visualizations for performance evaluation
- Bootstrap toggle for Random Forest (bagging vs. pasting)

![Example of binary classification of simple web app ](pic/Logistic Regression.png)


The dataset has 8124 samples with 22 categorical features, encoded numerically. Target: Edible (0) vs. Poisonous (1).

# Data Preprocessing

- All features are label-encoded (categorical to numeric).
- 70/30 train-test split with stratification.
- Note: Column name for target is 'class' (edible/poisonous). Update if your CSV uses 'type'.

## Classifiers and Hyperparameters

- **SVM**: C (regularization), kernel (rbf/linear), gamma (scale/auto).
- **Logistic Regression**: C, max iterations.
- **Random Forest**: Number of trees, max depth, min samples split/leaf, bootstrap (yes/no for bagging/pasting).

Models achieve near-perfect accuracy on this dataset due to strong feature correlations (e.g., odor is a key predictor).