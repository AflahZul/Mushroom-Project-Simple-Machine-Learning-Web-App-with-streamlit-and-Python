# Mushroom Classification Web App

A simple Streamlit app for binary classification: Predict if mushrooms are edible or poisonous using machine learning models.

## Description

This app uses the Mushrooms (fake) dataset to train and evaluate classifiers like SVM, Logistic Regression, and Random Forest. Users can tweak hyperparameters via the sidebar and visualize metrics (accuracy, precision, recall) along with plots like confusion matrices and ROC curves.

Key features:
- Interactive hyperparameter tuning
- Model comparison
- Visualizations for performance evaluation
- Bootstrap toggle for Random Forest (bagging vs. pasting)

The dataset has 8124 samples with 22 categorical features, encoded numerically. Target: Edible (0) vs. Poisonous (1).
