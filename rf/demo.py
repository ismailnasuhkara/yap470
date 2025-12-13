#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os, time
sys.path.append(os.getcwd())


# In[ ]:


import pandas as pd
import numpy as np

from decision_tree import DecisionTree
from random_forest import RandomForest
from metric_tracker import MetricTracker


# In[ ]:


from sklearn.model_selection import (
    KFold,
    train_test_split
)


# In[ ]:


dataset_path = "../dataset/diabetes_binary_5050split_health_indicators_BRFSS2023.csv"
dataset = pd.read_csv(dataset_path)
dataset.head()


# In[ ]:


y = dataset['Diabetes_binary']
X = dataset.drop(columns=['Diabetes_binary'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "rf_probs": RandomForest(store_probs=True),
    "rf_labels": RandomForest(store_probs=False),
    "dt_probs": DecisionTree(store_probs=True),
    "dt_labels": DecisionTree(store_probs=False)
}

metrics = {}

tracker = MetricTracker()


# In[ ]:


def cross_validate(model_name, model, X, y, kf):
    tracker.clear_cv_history()
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        row = tracker.compute_cv_metrics(model_name, model, X_val, y_val)
        row["fold"] = fold_idx
        tracker.add_cv_row(row)
    return tracker.cv_to_df()


# In[ ]:


# for model_name, model in models.items():
#     print(f"Cross validating {model_name}...")
#     start_time = time.time()

#     metrics[f"cv_{model_name}"] = cross_validate(model_name, model, X_train, y_train, kf)
    
#     runtime = time.time() - start_time
#     print(f"Cross validation finished in {runtime}\n") 


# In[ ]:


models["rf_probs"].fit(X_train, y_train)
row = tracker.compute_cv_metrics("rf_probs", models["rf_probs"], X_test, y_test)
tracker.add_cv_row(row)
tracker.cv_to_df()


# In[ ]:


metrics["cv_dt_probs"]


# In[ ]:


metrics["cv_rf_probs"]

