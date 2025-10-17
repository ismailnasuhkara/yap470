import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return left == None and right == None

class DecisionTree():
    def __init__(self, max_depth=2, min_samples_split=2):

    def _gini_index():
    
    def _entropy():

    def _information_gain():

    def _best_split():

    def fit():

    def predict():
