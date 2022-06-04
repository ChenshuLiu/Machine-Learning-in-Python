import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        # covariance matrix
        cov = np.cov(X.T)
        # eigenvector and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # sort eigenvectors with eigenvalues
        eigenvectors = eigenvectors.T
        # sort in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store only first n eigenvectors as new dimensions
        self.components = eigenvectors[0:self.n_components]
    
    # transform the data using new projection
    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)