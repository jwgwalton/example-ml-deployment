"""
Custom feature transformer for ML pipeline.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that creates polynomial features and interaction terms.
    
    This transformer:
    1. Computes squared features
    2. Computes interaction terms between features
    3. Normalizes the features to zero mean and unit variance
    """
    
    def __init__(self, add_squared=True, add_interactions=True):
        self.add_squared = add_squared
        self.add_interactions = add_interactions
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer by computing mean and std of the transformed features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        self : CustomFeatureTransformer
        """
        X_transformed = self._create_features(X)
        self.mean_ = np.mean(X_transformed, axis=0)
        self.std_ = np.std(X_transformed, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding polynomial and interaction features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_transformed : array-like
            Transformed data with additional features
        """
        X_transformed = self._create_features(X)
        # Normalize using fitted statistics
        X_transformed = (X_transformed - self.mean_) / self.std_
        return X_transformed
    
    def _create_features(self, X):
        """
        Internal method to create polynomial and interaction features.
        """
        X = np.array(X)
        features = [X]
        
        if self.add_squared:
            # Add squared features
            features.append(X ** 2)
        
        if self.add_interactions and X.shape[1] > 1:
            # Add pairwise interaction terms
            n_features = X.shape[1]
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            if interactions:
                features.extend(interactions)
        
        return np.hstack(features)
