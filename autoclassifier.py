#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class AutoClassifier:
    def __init__(self, models=None, cv=5, random_state=42):
        """Initialize with default models or custom models."""
        self.models = models or {
            "RandomForest": RandomForestClassifier(random_state=random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
            "SVM": SVC(random_state=random_state),
        }
        self.cv = cv
        self.best_model = None
        self.scaler = StandardScaler()
        self.random_state = random_state

    def fit(self, X, y):
        """Train the best model after hyperparameter tuning."""
        # Preprocess data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Evaluate models using cross-validation
        best_score = -1
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=self.cv)
            mean_score = np.mean(scores)
            print(f"{name} | Avg CV Accuracy: {mean_score:.3f}")

            if mean_score > best_score:
                best_score = mean_score
                self.best_model = model

        # Train the best model on full data
        self.best_model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, self.best_model.predict(X_val))
        print(f"\nBest Model: {type(self.best_model).__name__} | Validation Accuracy: {val_acc:.3f}")

    def predict(self, X):
        """Predict using the best model."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.scaler.transform(X)
        return self.best_model.predict(X)

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target

    clf = AutoClassifier()
    clf.fit(X, y)
    predictions = clf.predict(X[:5])
    print("\nSample Predictions:", predictions)


# In[ ]:




