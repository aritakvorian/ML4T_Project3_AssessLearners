import numpy as np
import random

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "atakvorian7"

    def study_group(self):
        return "atakvorian7"

    def add_evidence(self, data_x, data_y):
        self.X = np.array(data_x)
        self.Y = np.array(data_y).reshape(-1)
        self.tree = self._build_tree(self.X, self.Y)

    def _build_tree(self, X, Y):
        # Check leaf conditions
        if X.shape[0] <= self.leaf_size or np.all(Y == Y[0]):
            prediction = float(np.mean(Y))
            return np.array([[-1.0, prediction, np.nan, np.nan]])

        valid_features = []
        num_features = X.shape[1]

        for i in range(num_features):
            if not np.all(X[:, i] == X[0, i]):
                valid_features.append(i)

        # No split
        if len(valid_features) == 0:
            prediction = float(np.mean(Y))
            return np.array([[-1.0, prediction, np.nan, np.nan]])

        # Pick one feature at random
        best_feature = random.choice(valid_features)

        # Split on median of feature
        split_val = np.median(X[:, best_feature])
        left_mask = X[:, best_feature] <= split_val
        right_mask = X[:, best_feature] > split_val

        if not np.any(left_mask) or not np.any(right_mask):
            prediction = float(np.mean(Y))
            return np.array([[-1.0, prediction, np.nan, np.nan]])

        # Recursive call
        left_tree = self._build_tree(X[left_mask], Y[left_mask])
        right_tree = self._build_tree(X[right_mask], Y[right_mask])

        root = np.array([
            [float(best_feature), float(split_val), 1.0, float(left_tree.shape[0] + 1)]
        ])

        # Stack root + left subtree + right subtree
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        pts = np.array(points)
        if self.tree is None:
            raise ValueError("Learner has not been trained.")
        return np.array([self._query_point(p) for p in pts])

    def _query_point(self, p):
        node = 0
        while True:
            feat = int(self.tree[node, 0])
            if feat == -1:
                return float(self.tree[node, 1])
            split = self.tree[node, 1]
            if p[feat] <= split:
                node = node + int(self.tree[node, 2])
            else:
                node = node + int(self.tree[node, 3])
