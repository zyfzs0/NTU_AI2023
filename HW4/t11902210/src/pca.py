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
        #print(X.shape)
        self.mean = np.mean(X, axis=0)
        X_fit = X - self.mean
        #size = X_fit.shape[0]
        #X_cov = np.matmul(X_fit.T, X_fit)  / size
        X_cov =  np.cov(X_fit.T)
        # I consulted chatgpt about the usage of this np
        char_val, char_vec = np.linalg.eigh(X_cov) #***
        index = np.argsort(char_val)[::-1]
        # (how to calculate feature vectors)
        temp = char_vec[:,index]
        self.components = temp[:, 0 : self.n_components]
        #print(self.components.shape)


        #raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 1%
        # Hint: Use the calculated principal components to project the data onto a lower dimensional space
        X_tra = X - self.mean
        result = X_tra.dot(self.components)
        return result
        #raise NotImplementedError

    def reconstruct(self, X):
        #raise NotImplementedError
        #TODO: 1%
        # Hint: Use the calculated principal components to reconstruct the data back to its original space
        #X_re = X.dot(self.components.T)
        X_tra = X - self.mean
        X = X_tra.dot(self.components)
        X_re = X.dot(self.components.T)
        result = X_re + self.mean
        return result
