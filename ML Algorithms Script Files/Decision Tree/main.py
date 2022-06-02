import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

def accuracy(y_actual, y_pred):
    accuracy = np.sum(y_actual == y_pred) / len(y_actual)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

clf = DecisionTree(max_depth = 10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"The prediction accuracy of decision tree is: {accuracy(y_test, y_pred)}")