import numpy as np
import pandas as pd
from decision_tree import DecisionTree
from collections import Counter

class RandomForest():
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        store_probs: bool = False
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.store_probs = store_probs
        self.trees: list[DecisionTree] = []

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        elif self.max_features is None:
            return n_features
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def fit(self, X_pd: pd.DataFrame, y_pd: pd.Series):
        X = X_pd.to_numpy()
        y = y_pd.to_numpy()
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_feat_subsample = self._resolve_max_features(n_features)
        self.trees = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.randint(0, n_samples, size=n_samples)
                X_sample = X[idx]
                y_sample = y[idx]
            else:
                X_sample = X
                y_sample = y

            tree = DecisionTree(
                n_features=n_feat_subsample,
                max_depth=self.max_depth,
                min_samples_split=self.store_probs
            )
            tree.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
            self.trees.append(tree)

    def _predict_single(self, x: np.ndarray):
        votes = [tree._traverse_tree(x, tree.root) for tree in self.trees]
        
        if self.store_probs:
            votes = np.array(votes, dtype=float)
            return votes.mean(axis=0)
        else:
            counter = Counter(votes)
            return counter.most_common(1)[0][0]

    def predict(self, X_pd: pd.DataFrame) -> np.ndarray:
        X = X_pd.to_numpy()
        preds = []

        for x in X:
            # collect vote vectors from all trees
            votes = []
            for tree in self.trees:
                leaf_value = tree._traverse_tree(x, tree.root)  # count vector or prob vector
                leaf_value = np.array(leaf_value, dtype=float)
                votes.append(leaf_value)

            # average the vote vectors
            avg_vote = np.mean(votes, axis=0)

            # predicted class = argmax
            preds.append(self.classes[np.argmax(avg_vote)])

        return np.array(preds)

    def predict_proba(self, X_pd: pd.DataFrame) -> np.ndarray:
        X = X_pd.to_numpy()
        n = len(X)
        k = len(self.classes)

        results = np.zeros((n, k))


        for i, x in enumerate(X):
            votes = []
            for tree in self.trees:
                leaf_value = tree._traverse_tree(x, tree.root)

                leaf_value = np.array(leaf_value, dtype=float)
                s = leaf_value.sum()

                if s == 0:
                    # This should never happen if leaf stores count vectors correctly
                    raise ValueError(f"Leaf with zero count encountered: {leaf_value}")

                prob = leaf_value / s     # normalize counts OR prob vector
                votes.append(prob)

            # Average probabilities across trees
            results[i] = np.mean(votes, axis=0)

        return results

