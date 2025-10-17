from __future__ import annotations
import numpy as np

class Node():
    def __init__(self,
        feature_index: int | None = None,
        threshold: int | float = None,
        left: Node | None = None,
        right: Node | None = None,
        value: int | float = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTree():
    def __init__(
        self,
        n_features: int | None = None,
        max_depth: int = 2,
        min_samples_split: int = 10
    ):
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini_index(self, class_counts: list[int]) -> float:
    
    def _entropy(self, y: np.ndarray) -> float:

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[int | None, float | None, float]:

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0
    ) -> Node:

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int | float:

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

    def predict(self, X:np.ndarray) -> np.ndarray:

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
