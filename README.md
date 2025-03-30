# AutoML Classifier

A Python library for automated machine learning classification tasks. This package simplifies model selection, hyperparameter tuning, and evaluation with a scikit-learn compatible API.

## Features

- **Automatic model selection** from popular classifiers (Random Forest, Gradient Boosting, SVM)
- **Built-in cross-validation** for reliable performance estimation
- **Data preprocessing** including automatic feature scaling
- **Simple API** with familiar fit/predict interface
- **Pandas DataFrame support** for easy integration with existing workflows

## Installation

```bash
pip install ujwal_automl-classifier

## Basic Usage

from ujwal_automl-classifier import AutoClassifier
from sklearn.datasets import load_iris

# Load sample data
data = load_iris()
X, y = data.data, data.target

# Initialize and fit the classifier
clf = AutoClassifier()
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X[:5])
print(predictions)

## Advanced Usage

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

# Specify custom models to evaluate
models = {
    "ExtraTrees": ExtraTreesClassifier(),
    "NaiveBayes": GaussianNB()
}

clf = AutoClassifier(models=models)
clf.fit(X, y)

# After fitting
print(f"Best model: {clf.best_model}")
print(f"Validation accuracy: {clf.best_score:.3f}")

## API Reference

# AutoClassifier
AutoClassifier(models=None, cv=5, random_state=42)

models: Dictionary of models to evaluate (default: RandomForest, GradientBoosting, SVM)

cv: Number of cross-validation folds (default: 5)

random_state: Random seed for reproducibility

# Methods

fit(X, y): Train the classifier

predict(X): Make predictions


## License

This project is licensed under the MIT License