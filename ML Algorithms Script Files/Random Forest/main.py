import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from helper import accuracy
from random_forest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

clf = RandomForest(n_trees = 3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"The prediction accuracy of the random forest is: {accuracy(y_test, y_pred)}")