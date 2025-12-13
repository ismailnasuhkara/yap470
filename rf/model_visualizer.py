"""
Comprehensive visualization module for model evaluation.
Includes: Learning Curves, ROC Curves, Precision-Recall Curves,
Confusion Matrix, Feature Importance, and more.
"""

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
    """
    A comprehensive visualization class for machine learning model evaluation.
    """
    
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