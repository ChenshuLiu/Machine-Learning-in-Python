import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2)**2)

class KNN:
    def __init__(self, k = 3):
        assert (k >= 1) & (type(k) == int), f"k value {k} is invalid"
        self.k = k
        
    # fitting the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # to predict sample cases
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x):
        # compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples and labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote --> most common class label
        # the Counter function returns a tuple(value, # of occurrence)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]