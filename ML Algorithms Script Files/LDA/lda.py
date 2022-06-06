import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        # store the eigenvectors
        self.linear_discriminants = None
        
    # LDA is supervised, so input takes both X and y
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        # calculate the two scatter matrices
        mean_overall = np.mean(X, axis = 0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis = 0)
            # dimension computation: (4, n_c) * (n_c, 4) = (4, 4)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c) 
            
            n_c = X_c.shape[0]
            # dimension computation: (4, 1) * (1, 4) = (4, 4)
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * mean_diff.dot(mean_diff.T)
        
        A = np.linalg.inv(S_W).dot(S_B)
        
        # find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        self.linear_discriminants = eigenvectors[:self.n_components]
    
    def transform(self, X):
        # project data onto the new dimension components
        return np.dot(X, self.linear_discriminants.T)