import numpy as np
from sklearn import datasets

class SVM:
    def __init__(self, lr = 0.001, lambda_param = 0.1, n_iters = 1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        # format the target
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        # initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)