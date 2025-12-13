from __future__ import annotations  # to remove "" when type hinting Node
import numpy as np
import pandas as pd

class Node():
    def __init__(
        self,
        feature_index: int | None = None,
        threshold: int | float = None,
        left: Node | None = None,
        right: Node | None = None,
        value: int | float = None,
        store_probs: bool = False
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.store_probs = store_probs
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class DecisionTree():
    def __init__(
        self,
        n_features: int | None = None,
        max_depth: int = 2,
        min_samples_split: int = 10,
        store_probs: bool = False
    ):
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.store_probs = store_probs
        self.root = None

    def _gini_impurity(self, class_counts: list[int]) -> float:
        total = sum(class_counts)
        if total == 0:
            return 0.0
        probabilities = [count / total for count in class_counts]
        gini = 1.0 - sum(p**2 for p in probabilities)
        return gini

    def _entropy(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(p * np.log2(p) for p in probs if p > 0)
        return float(entropy)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[int | None, float | None, float]:
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None, 0.0

        best_feature, best_threshold = None, None
        best_impurity = float("inf")
        impurity_function = self._entropy if self.store_probs else self._gini_impurity

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                # Check for invalid split
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # Compute weighted impurity
                left_counts = np.unique(y_left, return_counts=True)[1]
                right_counts = np.unique(y_right, return_counts=True)[1]

                n_left = y_left.size
                n_right = y_right.size
                n_total = n_left + n_right

                if impurity_function == self._entropy:
                    left_impurity = self._entropy(y_left)
                    right_impurity = self._entropy(y_right)
                else:   # gini impurity
                    left_impurity = self._gini_impurity(left_counts)
                    right_impurity = self._gini_impurity(right_counts)

                weighted_impurity = (n_left/n_total) * left_impurity + (n_right/n_total) * right_impurity

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_impurity

    def _create_leaf_node(self, y: np.ndarray):
        count_vector = np.zeros(len(self.classes), dtype=int)
        values, counts = np.unique(y, return_counts=True)
        for v, c in zip(values, counts):
            idx = self.class_to_index[v]
            count_vector[idx] = c
        if self.store_probs == True:
            probs = counts / counts.sum()
            return Node(value=probs, store_probs=True)
        else:
            return Node(value=count_vector, store_probs=False)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> Node:
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Base Case/Stopping Conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1
        ):
            return self._create_leaf_node(y)

        # Create subsets for recursion and find the best split
        n_features_to_consider = self.n_features or n_features
        
        feature_indices = np.random.choice(
            X.shape[1], self.n_features, replace=False
        )
        feature_index, threshold, impurity = self._best_split(X, y, feature_indices)

        # If no valid split is found
        if feature_index is None:
            return self._create_leaf_node(y)

        # Split the data according to _best_split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        # Recursive to build children
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return current node
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
            store_probs=self.store_probs
        )

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int | float:
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def fit(self, X_pd:pd.DataFrame , y_pd: pd.Series) -> None:
        X = X_pd.to_numpy()
        y = y_pd.to_numpy()
        self.classes = np.unique(y)
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
<<<<<<< HEAD

        n_samples, n_features = X.shape
        self.n_features = self.n_features or n_features     # eliminates n_features == None

        self.root = self._build_tree(X, y)

    def predict(self, X:pd.DataFrame) -> np.ndarray:
        X = X.to_numpy()
        predictions = [self._traverse_tree(x, self.root) for x in X]
        predictions = [np.argmax(p) for p in predictions]
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
=======

        n_samples, n_features = X.shape
        self.n_features = self.n_features or n_features     # eliminates n_features == None

        self.root = self._build_tree(X, y)

    def predict(self, X_pd:pd.DataFrame) -> np.ndarray:
        X = X_pd.to_numpy()
        predictions = [self._traverse_tree(x, self.root) for x in X]
        predictions = [np.argmax(p) for p in predictions]
        return np.array(predictions)

    def predict_proba(self, X_pd: pd.DataFrame) -> np.ndarray:
        X = X_pd.to_numpy()
>>>>>>> c503bcd (Added hyperparameter tuning. Plotting needs improvement)
        results = []
        for x in X:
            value = self._traverse_tree(x, self.root)

            # Case 1: store_probs == True
            if isinstance(value, (list, np.ndarray)):
                prob = np.asarray(value, dtype=float)
                prob = prob / prob.sum()    # normalization
            # Case 2: store.probs == False
            else:
                prob = np.zeros(len(self.classes))
                class_index = np.where(self.classes == value)[0][0]
                prob[class_index] = 1.0
            
            results.append(prob)

<<<<<<< HEAD
        return np.vstack(results)
=======
        return np.vstack(results)

    def get_params(self, deep=True):
        return {
            'n_features': self.n_features,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'store_probs': self.store_probs
        }

    def set_params(self, **params):
        """Set the parameters of this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
>>>>>>> c503bcd (Added hyperparameter tuning. Plotting needs improvement)
