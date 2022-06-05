from sklearn.datasets import make_blobs
from kmeans import KMeans 
import numpy as np

X, y = make_blobs(centers = 4, n_samples = 500, n_features = 2, shuffle = True, random_state = 1234)
clusters = len(np.unique(y))
k = KMeans(K = clusters, max_iters = 150, plot_steps = False)
y_pred = k.predict(X)
k.plot()