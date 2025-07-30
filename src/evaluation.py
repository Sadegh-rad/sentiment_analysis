"""
Model Evaluation Module

This module contains functions and classes for evaluating machine learning
models for sentiment analysis, including metrics calculation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Model evaluation class for sentiment analysis.
    
    This class provides comprehensive evaluation metrics and visualizations
    for binary classification models.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate a model using standard classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        # Convert to numeric if necessary
        y_true = pd.to_numeric(y_true, errors='coerce')
        y_pred = pd.to_numeric(y_pred, errors='coerce')
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_score_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_model_with_probabilities(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_proba: np.ndarray, model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate a model including probability-based metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        # Get basic metrics
        metrics = self.evaluate_model(y_true, y_pred, model_name)
        
        # Add AUC if probabilities are available
        try:
            if y_proba.ndim > 1:
                # For binary classification, use positive class probabilities
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba_positive)
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            metrics['auc_roc'] = None
        
        # Update stored results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = "Model") -> str:
        """
        Print detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            
        Returns:
            str: Classification report string
        """
        report = classification_report(y_true, y_pred)
        print(f"\n{model_name} Classification Report:")
        print("=" * 50)
        print(report)
        return report
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            model_name: str = "Model",
                            figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str], optional): Class labels
            model_name (str): Name of the model
            figsize (Tuple[int, int]): Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels or ['Negative', 'Positive'],
                    yticklabels=labels or ['Negative', 'Positive'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model",
                      figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            figsize (Tuple[int, int]): Figure size
        """
        # Handle probability array
        if y_proba.ndim > 1:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
        auc = roc_auc_score(y_true, y_proba_positive)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   model_name: str = "Model",
                                   figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Prediction probabilities
            model_name (str): Name of the model
            figsize (Tuple[int, int]): Figure size
        """
        # Handle probability array
        if y_proba.ndim > 1:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, figsize: Tuple[int, int] = (12, 8)) -> pd.DataFrame:
        """
        Compare evaluation results of multiple models.
        
        Args:
            figsize (Tuple[int, int]): Figure size for visualization
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_model first.")
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.evaluation_results).T
        
        # Create visualization
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.ravel()
            
            for i, metric in enumerate(available_metrics):
                if i < len(axes):
                    comparison_df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
                    axes[i].set_title(f'{metric.capitalize()} Comparison')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return comparison_df
    
    def evaluate_multiple_models(self, models_dict: Dict[str, Any],
                                X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models and compare their performance.
        
        Args:
            models_dict (Dict[str, Any]): Dictionary of trained models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = {}
        
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            metrics = self.evaluate_model(y_test, y_pred, model_name)
            
            # Print classification report
            self.print_classification_report(y_test, y_pred, model_name)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, model_name=model_name)
            
            # Try to get probabilities and plot ROC curve
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    metrics_with_proba = self.evaluate_model_with_probabilities(
                        y_test, y_pred, y_proba, model_name
                    )
                    self.plot_roc_curve(y_test, y_proba, model_name)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    self.plot_roc_curve(y_test, y_scores, model_name)
            except Exception as e:
                print(f"Could not plot ROC curve for {model_name}: {e}")
        
        # Compare all models
        return self.compare_models()
    
    def save_evaluation_results(self, filepath: str) -> None:
        """
        Save evaluation results to a CSV file.
        
        Args:
            filepath (str): Path to save the results
        """
        if not self.evaluation_results:
            print("No evaluation results to save.")
            return
        
        results_df = pd.DataFrame(self.evaluation_results).T
        results_df.to_csv(filepath)
        print(f"Evaluation results saved to {filepath}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            Tuple[str, float]: Best model name and its score
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available.")
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in self.evaluation_results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        
        return best_model, best_score
    
    def create_evaluation_summary(self) -> str:
        """
        Create a summary of all evaluation results.
        
        Returns:
            str: Evaluation summary
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        summary = "Model Evaluation Summary\n"
        summary += "=" * 50 + "\n\n"
        
        for model_name, metrics in self.evaluation_results.items():
            summary += f"{model_name}:\n"
            for metric, value in metrics.items():
                if value is not None:
                    summary += f"  {metric}: {value:.4f}\n"
            summary += "\n"
        
        # Best models by different metrics
        key_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        summary += "Best Models by Metric:\n"
        summary += "-" * 30 + "\n"
        
        for metric in key_metrics:
            try:
                best_model, best_score = self.get_best_model(metric)
                summary += f"{metric}: {best_model} ({best_score:.4f})\n"
            except:
                pass
        
        return summary
