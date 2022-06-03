import numpy as np
import decision_tree

# help randomize sample for each tree
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size = n_samples, replace = True)
    return X[idxs], y[idxs]

class RandomForest:
    def __init__(self, n_trees = 100, min_samples_split = 2, max_depth = 100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = decision_tree.DecisionTree(min_samples_split = self.min_samples_split,
                               max_depth = self.max_depth,
                               n_feats = self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        # tree is the fitted decision tree model
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [decision_tree.most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)