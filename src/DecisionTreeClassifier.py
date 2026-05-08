from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    label: int | None = None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict(sample) for sample in X])

    def _grow_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        if n_labels == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(label=self._majority_class(y))

        if n_samples < self.min_samples_split:
            return Node(label=self._majority_class(y))

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return Node(label=self._majority_class(y))

        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            return Node(label=self._majority_class(y))

        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._gini_index(X[:, feature], y, threshold)
                if gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = gini

        return best_feature, best_threshold

    def _split_data(self, X, y, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    def _gini_index(self, feature_values, y, threshold):
        y_left = y[feature_values <= threshold]
        y_right = y[feature_values > threshold]

        n_samples = len(y)
        n_samples_left = len(y_left)
        n_samples_right = len(y_right)

        if n_samples_left == 0 or n_samples_right == 0:
            return float("inf")

        gini_left = 1 - np.sum((np.bincount(y_left) / n_samples_left) ** 2)
        gini_right = 1 - np.sum((np.bincount(y_right) / n_samples_right) ** 2)
        weight = n_samples_left / n_samples
        return weight * gini_left + (1 - weight) * gini_right

    def _predict(self, sample):
        node = self.root
        while node.left and node.right:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def _majority_class(self, y):
        return np.argmax(np.bincount(y))

    def print_tree(self, node, depth=0):
        if node.label is not None:
            print("  " * depth + f"Leaf Node: {node.label}")
            return

        print("  " * depth + f"Feature {node.feature} <= {node.threshold}")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)
