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
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 1%
        raise NotImplementedError

    def reconstruct(self, X):
        raise NotImplementedError
        #TODO: 1%
