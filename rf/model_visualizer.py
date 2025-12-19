import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve, validation_curve
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ModelVisualizer:
    def __init__(self, figsize=(10, 6), style='seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size for plots
        style : str
            Matplotlib style to use
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color palette
        self.colors = plt.cm.Set2.colors
    
    def plot_learning_curve(
        self,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'f1',
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        n_jobs: int = -1,
        title: Optional[str] = None
    ):
        """
        Plot learning curve showing training and validation scores vs training size.
        
        Parameters:
        -----------
        estimator : object
            The model to evaluate
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')
        train_sizes : array-like
            Relative or absolute numbers of training examples
        n_jobs : int
            Number of jobs to run in parallel (-1 uses all cores)
        title : str, optional
            Custom title for the plot
        """
        print("Computing learning curve...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y,
            cv=cv,
            scoring=scoring,
            train_sizes=train_sizes,
            n_jobs=n_jobs,
            random_state=42
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Training score
        ax.plot(train_sizes, train_mean, 'o-', color=self.colors[0], 
                label='Training score', linewidth=2, markersize=8)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=self.colors[0])
        
        # Validation score
        ax.plot(train_sizes, val_mean, 'o-', color=self.colors[1],
                label='Cross-validation score', linewidth=2, markersize=8)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color=self.colors[1])
        
        ax.set_xlabel('Training Examples', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Learning Curve ({scoring.upper()})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for final scores
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        ax.annotate(f'Final Train: {final_train:.3f}',
                   xy=(train_sizes[-1], final_train),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc=self.colors[0], alpha=0.3),
                   fontsize=9)
        ax.annotate(f'Final Val: {final_val:.3f}',
                   xy=(train_sizes[-1], final_val),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc=self.colors[1], alpha=0.3),
                   fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nLearning Curve Summary:")
        print(f"Final Training Score: {final_train:.4f} (±{train_std[-1]:.4f})")
        print(f"Final Validation Score: {final_val:.4f} (±{val_std[-1]:.4f})")
        print(f"Gap (Overfitting): {final_train - final_val:.4f}")
    
    def plot_overfitting_analysis(
        self,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'f1',
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        n_jobs: int = -1,
        title: Optional[str] = None
    ):
        """
        Comprehensive overfitting/underfitting analysis with gap visualization.
        
        Parameters:
        -----------
        estimator : object
            The model to evaluate
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
        train_sizes : array-like
            Relative or absolute numbers of training examples
        n_jobs : int
            Number of jobs to run in parallel
        title : str, optional
            Custom title for the plot
        """
        print("Computing overfitting analysis...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y,
            cv=cv,
            scoring=scoring,
            train_sizes=train_sizes,
            n_jobs=n_jobs,
            random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        gap = train_mean - val_mean
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Training vs Validation with gap shading
        ax1.plot(train_sizes, train_mean, 'o-', color=self.colors[0], 
                label='Training score', linewidth=2, markersize=8)
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=self.colors[0])
        
        ax1.plot(train_sizes, val_mean, 'o-', color=self.colors[1],
                label='Validation score', linewidth=2, markersize=8)
        ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color=self.colors[1])
        
        # Shade the gap (overfitting region)
        ax1.fill_between(train_sizes, train_mean, val_mean, 
                        alpha=0.3, color='red', label='Overfitting Gap')
        
        ax1.set_xlabel('Training Examples', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'{scoring.upper()} Score', fontsize=12, fontweight='bold')
        ax1.set_title('Training vs Validation Score', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overfitting gap over training size
        ax2.plot(train_sizes, gap, 'o-', color='red', linewidth=2, markersize=8)
        ax2.fill_between(train_sizes, gap, 0, alpha=0.3, color='red')
        ax2.axhline(y=0.05, color='orange', linestyle='--', label='Warning threshold (0.05)')
        ax2.axhline(y=0.1, color='darkred', linestyle='--', label='Critical threshold (0.1)')
        
        ax2.set_xlabel('Training Examples', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Gap (Train - Val)', fontsize=12, fontweight='bold')
        ax2.set_title('Overfitting Gap Analysis', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Overfitting/Underfitting Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Diagnosis
        final_gap = gap[-1]
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        
        print(f"\n{'='*60}")
        print("OVERFITTING ANALYSIS")
        print(f"{'='*60}")
        print(f"Final Training Score:   {final_train:.4f} (±{train_std[-1]:.4f})")
        print(f"Final Validation Score: {final_val:.4f} (±{val_std[-1]:.4f})")
        print(f"Overfitting Gap:        {final_gap:.4f}")
        print(f"\nDiagnosis:")
        
        if final_gap < 0.02:
            print("✓ Excellent! Model is well-balanced.")
        elif final_gap < 0.05:
            print("✓ Good. Minor overfitting detected.")
        elif final_gap < 0.1:
            print("⚠ Warning! Moderate overfitting detected.")
            print("  Recommendations: Increase regularization, reduce model complexity")
        else:
            print("✗ Critical! Severe overfitting detected.")
            print("  Recommendations: Significantly reduce model complexity,")
            print("  increase regularization, or collect more training data")
        
        if final_val < 0.6:
            print("\n⚠ Underfitting also detected (low validation score).")
            print("  Recommendations: Increase model complexity or add features")
        
        print(f"{'='*60}\n")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        title: Optional[str] = None
    ):
        """
        Plot training/validation/test metrics over epochs or iterations.
        
        Parameters:
        -----------
        history : dict
            Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
            Each value should be a list of scores per epoch/iteration.
        metrics : list, optional
            List of metric names to plot. If None, plots all available.
        title : str, optional
            Custom title for the plot
        
        Example usage:
        --------------
        history = {
            'train_loss': [0.5, 0.4, 0.3, 0.2],
            'val_loss': [0.6, 0.5, 0.45, 0.4],
            'train_acc': [0.7, 0.8, 0.85, 0.9],
            'val_acc': [0.65, 0.75, 0.8, 0.85]
        }
        viz.plot_training_history(history)
        """
        if not history:
            print("No training history provided.")
            return
        
        # Auto-detect metrics if not specified
        if metrics is None:
            # Group by metric type (loss, acc, f1, etc.)
            metric_types = set()
            for key in history.keys():
                # Extract metric type (e.g., 'loss' from 'train_loss')
                metric_type = key.replace('train_', '').replace('val_', '').replace('test_', '')
                metric_types.add(metric_type)
            metrics = list(metric_types)
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Plot train, val, test for this metric
            for prefix, color, marker in [('train', self.colors[0], 'o'), 
                                          ('val', self.colors[1], 's'), 
                                          ('test', self.colors[2], '^')]:
                key = f"{prefix}_{metric}"
                if key in history:
                    epochs = range(1, len(history[key]) + 1)
                    ax.plot(epochs, history[key], marker=marker, 
                           label=f"{prefix.capitalize()} {metric}", 
                           color=color, linewidth=2, markersize=6)
            
            ax.set_xlabel('Epoch / Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.upper()} over Time', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_error_per_epoch(
        self,
        train_errors: List[float],
        val_errors: Optional[List[float]] = None,
        test_errors: Optional[List[float]] = None,
        error_type: str = 'loss',
        title: Optional[str] = None
    ):
        """
        Plot training/validation/test error per epoch.
        
        Parameters:
        -----------
        train_errors : list
            Training errors per epoch
        val_errors : list, optional
            Validation errors per epoch
        test_errors : list, optional
            Test errors per epoch (typically evaluated once at the end)
        error_type : str
            Type of error ('loss', 'error_rate', 'mse', etc.)
        title : str, optional
            Custom title for the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_errors) + 1)
        
        ax.plot(epochs, train_errors, 'o-', color=self.colors[0],
               label=f'Training {error_type}', linewidth=2, markersize=6)
        
        if val_errors is not None:
            ax.plot(epochs, val_errors, 's-', color=self.colors[1],
                   label=f'Validation {error_type}', linewidth=2, markersize=6)
        
        if test_errors is not None:
            ax.plot(epochs, test_errors, '^-', color=self.colors[2],
                   label=f'Test {error_type}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Epoch / Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel(error_type.upper(), fontsize=12, fontweight='bold')
        ax.set_title(title or f'{error_type.upper()} per Epoch', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Highlight minimum validation error if available
        if val_errors is not None:
            min_idx = np.argmin(val_errors)
            min_val = val_errors[min_idx]
            ax.axvline(x=min_idx+1, color='red', linestyle='--', alpha=0.5)
            ax.annotate(f'Best Val\nEpoch: {min_idx+1}\nError: {min_val:.4f}',
                       xy=(min_idx+1, min_val),
                       xytext=(20, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_validation_curve(
        self,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_name: str,
        param_range: List[Any],
        cv: int = 5,
        scoring: str = 'f1',
        n_jobs: int = -1,
        title: Optional[str] = None
    ):
        """
        Plot validation curve showing training and validation scores vs parameter value.
        
        Parameters:
        -----------
        estimator : object
            The model to evaluate
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        param_name : str
            Name of parameter to vary
        param_range : list
            Values of the parameter to test
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
        n_jobs : int
            Number of jobs to run in parallel
        title : str, optional
            Custom title for the plot
        """
        print(f"Computing validation curve for {param_name}...")
        
        train_scores, val_scores = validation_curve(
            estimator, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(param_range, train_mean, 'o-', color=self.colors[0],
                label='Training score', linewidth=2, markersize=8)
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=self.colors[0])
        
        ax.plot(param_range, val_mean, 'o-', color=self.colors[1],
                label='Cross-validation score', linewidth=2, markersize=8)
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color=self.colors[1])
        
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Validation Curve ({param_name})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Mark best parameter
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]
        ax.axvline(best_param, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.annotate(f'Best: {best_param}\nScore: {best_score:.3f}',
                   xy=(best_param, best_score),
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                   fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nValidation Curve Summary:")
        print(f"Best {param_name}: {best_param}")
        print(f"Best Validation Score: {best_score:.4f} (±{val_std[best_idx]:.4f})")
    
    def plot_hyperparameter_heatmap(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        score_col: str = 'mean_test_score',
        title: Optional[str] = None
    ):
        """
        Plot heatmap of hyperparameter combinations (for GridSearchCV results).
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame from GridSearchCV.cv_results_
        param1 : str
            First parameter name (will be x-axis)
        param2 : str
            Second parameter name (will be y-axis)
        score_col : str
            Column name for the score to plot
        title : str, optional
            Custom title for the plot
        """
        # Extract relevant columns
        param1_col = f'param_{param1}'
        param2_col = f'param_{param2}'
        
        if param1_col not in results_df.columns or param2_col not in results_df.columns:
            print(f"Error: Parameters {param1} or {param2} not found in results.")
            return
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            values=score_col,
            index=param2_col,
            columns=param1_col,
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': score_col}, ax=ax, linewidths=0.5)
        
        ax.set_xlabel(param1, fontsize=12, fontweight='bold')
        ax.set_ylabel(param2, fontsize=12, fontweight='bold')
        ax.set_title(title or f'Hyperparameter Heatmap: {param1} vs {param2}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        title: Optional[str] = None
    ):
        """
        Plot ROC curves for one or multiple models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of {model_name: fitted_model}
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels (binary)
        title : str, optional
            Custom title for the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (name, model) in enumerate(models.items()):
            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    y_score = y_proba[:, 1]
                else:
                    y_score = y_proba[:, 0]
            else:
                print(f"Warning: {name} does not have predict_proba method")
                continue
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})',
                   color=self.colors[idx % len(self.colors)])
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title or 'ROC Curve Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        title: Optional[str] = None
    ):
        """
        Plot Precision-Recall curves for one or multiple models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of {model_name: fitted_model}
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels (binary)
        title : str, optional
            Custom title for the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    y_score = y_proba[:, 1]
                else:
                    y_score = y_proba[:, 0]
            else:
                print(f"Warning: {name} does not have predict_proba method")
                continue
            
            # Compute PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            avg_precision = average_precision_score(y_test, y_score)
            
            # Plot
            ax.plot(recall, precision, linewidth=2,
                   label=f'{name} (AP = {avg_precision:.3f})',
                   color=self.colors[idx % len(self.colors)])
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Precision-Recall Curve Comparison',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        labels: Optional[List[str]] = None,
        normalize: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        model : object
            Fitted model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        labels : list, optional
            Class labels for display
        normalize : str, optional
            'true', 'pred', 'all' or None
        title : str, optional
            Custom title
        """
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, normalize=normalize)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=True, square=True,
                   xticklabels=labels or ['Class 0', 'Class 1'],
                   yticklabels=labels or ['Class 0', 'Class 1'],
                   ax=ax, linewidths=1, linecolor='gray')
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=labels))
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: Optional[str] = None
    ):
        """
        Plot feature importance bar chart.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        importances : array-like
            Feature importance scores
        top_n : int
            Number of top features to display
        title : str, optional
            Custom title
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        ax.barh(range(len(indices)), importances[indices],
               color=self.colors[0], alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Top {top_n} Feature Importances',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_curve(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_bins: int = 10,
        title: Optional[str] = None
    ):
        """
        Plot calibration curve (reliability diagram).
        
        Parameters:
        -----------
        models : dict
            Dictionary of {model_name: fitted_model}
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels (binary)
        n_bins : int
            Number of bins for calibration
        title : str, optional
            Custom title
        """
        from sklearn.calibration import calibration_curve
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    y_score = y_proba[:, 1]
                else:
                    y_score = y_proba[:, 0]
            else:
                continue
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_score, n_bins=n_bins, strategy='uniform'
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-',
                   linewidth=2, label=name, color=self.colors[idx % len(self.colors)])
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
        title: Optional[str] = None
    ):
        """
        Plot bar chart comparing multiple models across metrics.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with 'model' column and metric columns
        metrics : list
            List of metric names to compare
        title : str, optional
            Custom title
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in results_df.columns:
                ax = axes[idx]
                models = results_df['model'].values
                scores = results_df[metric].values
                
                bars = ax.bar(range(len(models)), scores,
                             color=[self.colors[i % len(self.colors)] for i in range(len(models))],
                             alpha=0.8, edgecolor='black')
                
                # Highlight best
                best_idx = np.argmax(scores)
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('red')
                bars[best_idx].set_linewidth(2)
                
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel('Score', fontsize=11, fontweight='bold')
                ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 1.05])
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle(title or 'Model Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_grid_search_results(
        self,
        cv_results: Dict[str, Any],
        param_name: str,
        top_n: int = 10,
        title: Optional[str] = None
    ):
        """
        Visualize GridSearchCV results for a specific parameter.
        
        Parameters:
        -----------
        cv_results : dict
            cv_results_ from GridSearchCV
        param_name : str
            Parameter name to visualize
        top_n : int
            Number of top configurations to show
        title : str, optional
            Custom title
        """
        results_df = pd.DataFrame(cv_results)
        
        # Filter by parameter if it varies
        param_col = f'param_{param_name}'
        if param_col not in results_df.columns:
            print(f"Parameter {param_name} not found in results.")
            return
        
        # Group by parameter value and aggregate
        grouped = results_df.groupby(param_col).agg({
            'mean_test_score': 'mean',
            'std_test_score': 'mean',
            'mean_train_score': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Score vs Parameter
        ax1.errorbar(grouped[param_col], grouped['mean_test_score'],
                    yerr=grouped['std_test_score'], marker='o', 
                    linewidth=2, capsize=5, label='Test Score')
        ax1.plot(grouped[param_col], grouped['mean_train_score'],
                marker='s', linewidth=2, label='Train Score')
        ax1.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Score vs {param_name}', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Top N configurations
        top_configs = results_df.nsmallest(top_n, 'rank_test_score')
        x_pos = np.arange(len(top_configs))
        ax2.bar(x_pos, top_configs['mean_test_score'],
               yerr=top_configs['std_test_score'],
               capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Test Score', fontsize=12, fontweight='bold')
        ax2.set_title(f'Top {top_n} Configurations', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(range(1, len(top_configs) + 1))
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title or 'Grid Search Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_cross_validation_scores(
        self,
        cv_scores: np.ndarray,
        metric_name: str = 'Score',
        title: Optional[str] = None
    ):
        """
        Visualize cross-validation scores across folds.
        
        Parameters:
        -----------
        cv_scores : array-like
            Cross-validation scores for each fold
        metric_name : str
            Name of the metric
        title : str, optional
            Custom title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = range(1, len(cv_scores) + 1)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Plot 1: Scores per fold
        ax1.bar(folds, cv_scores, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax1.axhline(y=mean_score, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax1.fill_between(folds, mean_score - std_score, mean_score + std_score,
                        alpha=0.2, color='red', label=f'±1 Std: {std_score:.4f}')
        ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax1.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax1.set_title(f'{metric_name} per Fold', fontsize=13, fontweight='bold')
        ax1.set_xticks(folds)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Box plot
        ax2.boxplot([cv_scores], labels=[metric_name], patch_artist=True,
                   boxprops=dict(facecolor=self.colors[1], alpha=0.7))
        ax2.scatter([1] * len(cv_scores), cv_scores, color='red', 
                   s=50, zorder=3, alpha=0.6)
        ax2.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax2.set_title('Score Distribution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title or f'Cross-Validation Results ({len(cv_scores)} Folds)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(f"\nCross-Validation Summary:")
        print(f"Mean {metric_name}: {mean_score:.4f}")
        print(f"Std {metric_name}:  {std_score:.4f}")
        print(f"Min {metric_name}:  {np.min(cv_scores):.4f}")
        print(f"Max {metric_name}:  {np.max(cv_scores):.4f}")

    def plot_randomized_search_results(
        self,
        cv_results: Dict[str, Any],
        param_name: str,
        top_n: int = 15,
        title: Optional[str] = None
    ):
        """
        Visualize RandomizedSearchCV results with scatter plot and parameter importance.
        
        Parameters:
        -----------
        cv_results : dict
            cv_results_ from RandomizedSearchCV
        param_name : str
            Primary parameter name to visualize on x-axis
        top_n : int
            Number of top configurations to highlight
        title : str, optional
            Custom title
        """
        results_df = pd.DataFrame(cv_results)
        
        # Check if parameter exists
        param_col = f'param_{param_name}'
        if param_col not in results_df.columns:
            print(f"Parameter {param_name} not found in results.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Score vs Parameter (Scatter)
        ax1 = axes[0, 0]
        x_vals = results_df[param_col].values
        y_vals = results_df['mean_test_score'].values
        colors_map = results_df['rank_test_score'].values
        
        scatter = ax1.scatter(x_vals, y_vals, 
                            c=colors_map, cmap='RdYlGn_r', 
                            s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Highlight best configuration
        best_idx = results_df['rank_test_score'].idxmin()
        best_x = results_df.loc[best_idx, param_col]
        best_y = results_df.loc[best_idx, 'mean_test_score']
        ax1.scatter(best_x, best_y, s=300, marker='*', 
                color='gold', edgecolors='red', linewidth=2, 
                label='Best Config', zorder=5)
        
        ax1.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Test Score vs {param_name}', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Rank', fontsize=10, fontweight='bold')
        
        # Plot 2: Train vs Test Score Comparison
        ax2 = axes[0, 1]
        train_scores = results_df['mean_train_score'].values
        test_scores = results_df['mean_test_score'].values
        
        ax2.scatter(train_scores, test_scores, 
                c=colors_map, cmap='RdYlGn_r',
                s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Best config
        best_train = results_df.loc[best_idx, 'mean_train_score']
        best_test = results_df.loc[best_idx, 'mean_test_score']
        ax2.scatter(best_train, best_test, s=300, marker='*',
                color='gold', edgecolors='red', linewidth=2, zorder=5)
        
        # Diagonal line (perfect fit)
        min_val = min(train_scores.min(), test_scores.min())
        max_val = max(train_scores.max(), test_scores.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'k--', linewidth=2, alpha=0.5, label='Perfect Fit')
        
        ax2.set_xlabel('Train Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Test Score', fontsize=12, fontweight='bold')
        ax2.set_title('Train vs Test Score (Overfitting Check)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top N Configurations
        ax3 = axes[1, 0]
        top_configs = results_df.nsmallest(top_n, 'rank_test_score')
        x_pos = np.arange(len(top_configs))
        
        bars = ax3.bar(x_pos, top_configs['mean_test_score'],
                    yerr=top_configs['std_test_score'],
                    capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color bars by rank
        for i, (bar, rank) in enumerate(zip(bars, top_configs['rank_test_score'])):
            if rank == 1:
                bar.set_color('gold')
                bar.set_edgecolor('red')
                bar.set_linewidth(2.5)
            elif rank <= 3:
                bar.set_color(self.colors[1])
            else:
                bar.set_color(self.colors[0])
        
        ax3.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Test Score', fontsize=12, fontweight='bold')
        ax3.set_title(f'Top {top_n} Configurations', fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(top_configs['rank_test_score'].values)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add score annotations
        for i, (x, y) in enumerate(zip(x_pos, top_configs['mean_test_score'])):
            if i < 3:  # Annotate top 3
                ax3.text(x, y, f'{y:.4f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 4: Score Distribution
        ax4 = axes[1, 1]
        
        # Histogram
        ax4.hist(results_df['mean_test_score'], bins=20, 
                alpha=0.7, color=self.colors[0], edgecolor='black', linewidth=1.5)
        
        # Add vertical lines for statistics
        mean_score = results_df['mean_test_score'].mean()
        median_score = results_df['mean_test_score'].median()
        best_score = results_df['mean_test_score'].max()
        
        ax4.axvline(mean_score, color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax4.axvline(median_score, color='green', linestyle='--', 
                linewidth=2, label=f'Median: {median_score:.4f}')
        ax4.axvline(best_score, color='red', linestyle='--', 
                linewidth=2, label=f'Best: {best_score:.4f}')
        
        ax4.set_xlabel('Test Score', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Score Distribution', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title or 'Randomized Search Results', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print("RANDOMIZED SEARCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total Configurations Tested: {len(results_df)}")
        print(f"Best Score: {best_score:.4f}")
        print(f"Mean Score: {mean_score:.4f} (±{results_df['mean_test_score'].std():.4f})")
        print(f"Median Score: {median_score:.4f}")
        print(f"Score Range: [{results_df['mean_test_score'].min():.4f}, {results_df['mean_test_score'].max():.4f}]")
        print(f"\nBest Parameters:")
        for param, value in results_df.loc[best_idx, 'params'].items():
            print(f"  {param}: {value}")
        print(f"{'='*60}\n")


    def plot_cross_validation_comparison(
        self,
        cv_results_dict: Dict[str, np.ndarray],
        metric_name: str = 'F1 Score',
        title: Optional[str] = None
    ):
        """
        Compare cross-validation scores across multiple models or configurations.
        
        Parameters:
        -----------
        cv_results_dict : dict
            Dictionary of {model_name: cv_scores_array}
        metric_name : str
            Name of the metric being compared
        title : str, optional
            Custom title
        """
        n_models = len(cv_results_dict)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare data
        model_names = list(cv_results_dict.keys())
        all_scores = list(cv_results_dict.values())
        means = [np.mean(scores) for scores in all_scores]
        stds = [np.std(scores) for scores in all_scores]
        
        # Plot 1: Bar chart with error bars
        ax1 = axes[0]
        x_pos = np.arange(len(model_names))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=8, 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(self.colors[i % len(self.colors)])
        
        # Highlight best
        best_idx = np.argmax(means)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2.5)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax1.set_title(f'Mean {metric_name} with Std Dev', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.4f}\n±{std:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Box plot comparison
        ax2 = axes[1]
        bp = ax2.boxplot(all_scores, labels=model_names, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Color boxes
        for i, box in enumerate(bp['boxes']):
            if i == best_idx:
                box.set_facecolor('gold')
                box.set_edgecolor('red')
                box.set_linewidth(2.5)
            else:
                box.set_facecolor(self.colors[i % len(self.colors)])
                box.set_alpha(0.7)
        
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax2.set_title('Score Distribution (Box Plot)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Scores per fold (line plot)
        ax3 = axes[2]
        max_folds = max(len(scores) for scores in all_scores)
        folds = range(1, max_folds + 1)
        
        for i, (name, scores) in enumerate(cv_results_dict.items()):
            if i == best_idx:
                ax3.plot(range(1, len(scores) + 1), scores, 
                        marker='o', linewidth=3, markersize=10,
                        label=name, color='gold', 
                        markeredgecolor='red', markeredgewidth=2)
            else:
                ax3.plot(range(1, len(scores) + 1), scores,
                        marker='o', linewidth=2, markersize=8,
                        label=name, color=self.colors[i % len(self.colors)])
        
        ax3.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax3.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax3.set_title('Scores Across Folds', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(folds)
        
        plt.suptitle(title or f'Cross-Validation {metric_name} Comparison',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print comparison summary
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION COMPARISON - {metric_name}")
        print(f"{'='*60}")
        for i, (name, scores) in enumerate(cv_results_dict.items()):
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            min_val = np.min(scores)
            max_val = np.max(scores)
            star = " ⭐ BEST" if i == best_idx else ""
            print(f"\n{name}{star}")
            print(f"  Mean: {mean_val:.4f} (±{std_val:.4f})")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  Fold Scores: {[f'{s:.4f}' for s in scores]}")
            print(f"{'='*60}\n")