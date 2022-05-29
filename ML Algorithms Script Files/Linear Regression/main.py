import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from lr import LinearRegression
# data generation
X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

regressor = LinearRegression(lr = 0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
print(f"The prediction accuracy, according to MSE, is {MSE(y_test, predicted)}")