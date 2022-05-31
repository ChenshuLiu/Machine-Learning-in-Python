import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

def accuracy(y_actual, y_pred):
    accuracy = np.sum(y_actual == y_pred) / len(y_actual)
    return accuracy

X, y = datasets.make_blobs(n_samples = 150, n_features = 2, centers = 2, cluster_std = 1.05, random_state = 123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

p = Perceptron(learning_rate  = 0.01, n_iters = 1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)
print(f"The prediction accuracy of the perceptron is {accuracy(y_test, predictions)}")