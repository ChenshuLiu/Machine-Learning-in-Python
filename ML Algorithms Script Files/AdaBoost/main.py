from helper import accuracy
from adaboost import Adaboost
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
X = data.data
y = data.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

clf = Adaboost(n_clf = 5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"The prediction accuracy of the AdaBoost algorithm is: {accuracy(y_test, y_pred)}")