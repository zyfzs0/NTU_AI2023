import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        # Hint: Use existing method to calculate covariance matrix and its eigenvalues and eigenvectors
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        # Hint: Use the calculated principal components to project the data onto a lower dimensional space
        raise NotImplementedError

    def reconstruct(self, X):
        #TODO: 2%
        # Hint: Use the calculated principal components to reconstruct the data back to its original space
        raise NotImplementedError
