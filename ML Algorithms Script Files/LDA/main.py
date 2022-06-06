from lda import LDA
import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target
lda = LDA(2)
lda.fit(X, y)
X_projected = lda.transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
plt.scatter(x1, x2,
           c = y, edgecolor = "none", alpha = 0.8,
           cmap = plt.cm.get_cmap('viridis', 3))
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.colorbar()
plt.show()