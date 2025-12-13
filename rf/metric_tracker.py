import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
from typing import Optional, Sequence, Dict, Any

class MetricTracker:
    DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "roc", "logloss"]

    def __init__(self, metrics: Optional[Sequence[str]] = None, average: str = "weighted"):
        self.metrics = list(metrics) if metrics is not None else list(self.DEFAULT_METRICS)
        self.average = average
        self.cv_history: list[Dict[str, Any]] = []
        self.tuning_history: list[Dict[str, Any]] = []
        self.test_history: Optional[Dict[str, Any]] = None

    def _compute_metrics(self, y_true: pd.Series, y_pred: Optional[pd.Series], y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if "accuracy" in self.metrics:
            out["accuracy"] = float(accuracy_score(y_true, y_pred))

        if "precision" in self.metrics:
            out["precision"] = float(precision_score(y_true, y_pred, average=self.average, zero_division=0))

        if "recall" in self.metrics:
            out["recall"] = float(recall_score(y_true, y_pred, average=self.average, zero_division=0))

        if "f1" in self.metrics:
            out["f1"] = float(f1_score(y_true, y_pred, average=self.average, zero_division=0))

        if "roc" in self.metrics:
            roc_val = None
            if y_prob is not None:
                try:
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        roc_val = float(roc_auc_score(y_true, y_prob[:, 1]))
                except Exception:
                    roc_val = None
            out["roc"] = roc_val

        if "logloss" in self.metrics:
            ll = None
            if y_prob is not None:
                try:
                    # For binary, ensure shape (n, 2) or (n,)
                    if y_prob.ndim == 1:
                        # convert to two-column probability: p and 1-p
                        p = y_prob
                        y_prob_2 = np.vstack([1 - p, p]).T
                        ll = float(log_loss(y_true, y_prob_2))
                    else:
                        ll = float(log_loss(y_true, y_prob))
                except Exception:
                    ll = None
            out["logloss"] = ll

        return out

    def _safe_predict_proba(self, model, X) -> Optional[np.ndarray]:
        if not hasattr(model, "predict_proba"):
            return None
        try:
            return np.asarray(model.predict_proba(X), dtype=float)
        except Exception as e:
            return None

    def compute_cv_metrics(self, model_name:str, model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        y_pred = model.predict(X_val)
        y_prob = self._safe_predict_proba(model, X_val)
        metrics = self._compute_metrics(y_val, y_pred, y_prob)
        row = {"model": model_name}
        row.update(metrics)
        return row

    def add_cv_row(self, row: Dict[str, Any]) -> None:
        self.cv_history.append(row)

    def clear_cv_history(self) -> None:
        self.cv_history = []

    def cv_to_df(self, refit: str = "recall") -> pd.DataFrame:
        out = pd.DataFrame(self.cv_history) if self.cv_history else pd.DataFrame()
        out = out.sort_values(by=refit, ascending=False)
        return out
        

    def add_tuning_row(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        row = dict(params)
        row.update(metrics)
        self.tuning_history.append(row)

    def tuning_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.tuning_history) if self.tuning_history else pd.DataFrame()

    def plot_tuning_metrics(self, x_param: str, metric_names: Optional[Sequence[str]] = None):
        df = self.tuning_to_df()
        if df.empty:
            print("No tuning history to plot.")
            return
        metric_names = metric_names or [m for m in self.metrics if m in df.columns]
        plt.figure(figsize=(10, 6))
        for metric in metric_names:
            plt.plot(df[x_param], df[metric], marker='o', label=metric)
        plt.xlabel(x_param)
        plt.ylabel("Score")
        plt.title("Hyperparameter Tuning Metrics")
        plt.legend()
        plt.grid(True)
        plt.show()

    def compute_test_metrics(self, model_name: str, model, X_test, y_test) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        y_prob = self._safe_predict_proba(model, X_test)
        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        row = {"model": model_name}
        row.update(metrics)
        self.test_history = row
        return row

    def test_to_df(self, refit: str = "recall") -> pd.DataFrame:
        out = pd.DataFrame([self.test_history]) if self.test_history else pd.DataFrame()
        out = out.sort_values(by=refit, ascending=False)
        return out
