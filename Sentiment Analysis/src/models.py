"""
Machine Learning Models Module

This module contains sentiment classification models including 
Support Vector Machine, Logistic Regression, and Naive Bayes classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Any, Optional, Tuple, List
import joblib
import warnings

warnings.filterwarnings('ignore')


class SentimentClassifier:
    """
    Sentiment classification class with multiple ML models.
    
    This class provides a unified interface for training and evaluating
    different machine learning models for sentiment analysis.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sentiment classifier.
        
        Args:
            random_state (int): Random state for reproducible results
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all machine learning models."""
        self.models = {
            'linear_svm': LinearSVC(random_state=self.random_state),
            'logistic_regression': LogisticRegression(
                solver='saga', 
                fit_intercept=True, 
                random_state=self.random_state,
                max_iter=1000
            ),
            'naive_bayes': BernoulliNB()
        }
    
    def train_model(self, model_name: str, X_train, y_train) -> None:
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            X_train: Training features
            y_train: Training labels
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. "
                           f"Available models: {list(self.models.keys())}")
        
        print(f"Training {model_name}...")
        self.models[model_name].fit(X_train, y_train)
        print(f"{model_name} training completed.")
    
    def train_all_models(self, X_train, y_train) -> None:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
    
    def predict(self, model_name: str, X_test) -> np.ndarray:
        """
        Make predictions using a specific model.
        
        Args:
            model_name (str): Name of the model
            X_test: Test features
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available.")
        
        model = self.models[model_name]
        if not hasattr(model, 'predict'):
            raise ValueError(f"Model {model_name} not trained yet.")
        
        return model.predict(X_test)
    
    def predict_proba(self, model_name: str, X_test) -> np.ndarray:
        """
        Get prediction probabilities using a specific model.
        
        Args:
            model_name (str): Name of the model
            X_test: Test features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available.")
        
        model = self.models[model_name]
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} doesn't support probability prediction.")
        
        return model.predict_proba(X_test)
    
    def cross_validate_model(self, model_name: str, X_train, y_train, 
                           cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation on a specific model.
        
        Args:
            model_name (str): Name of the model
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            Dict: Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available.")
        
        model = self.models[model_name]
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def cross_validate_all_models(self, X_train, y_train, cv: int = 5, 
                                scoring: str = 'accuracy') -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            Dict: Cross-validation results for all models
        """
        cv_results = {}
        
        for model_name in self.models.keys():
            print(f"Cross-validating {model_name}...")
            cv_results[model_name] = self.cross_validate_model(
                model_name, X_train, y_train, cv, scoring
            )
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name: str, X_train, y_train, 
                            param_grid: Dict[str, List], cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model
            X_train: Training features
            y_train: Training labels
            param_grid (Dict): Parameter grid for grid search
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict: Best parameters and score
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available.")
        
        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def get_default_param_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get default parameter grids for hyperparameter tuning.
        
        Returns:
            Dict: Default parameter grids for each model
        """
        return {
            'linear_svm': {
                'C': [0.1, 1.0, 10.0],
                'loss': ['hinge', 'squared_hinge']
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        }
    
    def find_best_model(self, X_train, y_train, X_test, y_test, 
                       use_cross_validation: bool = True) -> Tuple[str, Dict[str, float]]:
        """
        Find the best performing model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            use_cross_validation (bool): Whether to use cross-validation
            
        Returns:
            Tuple: Best model name and its performance metrics
        """
        from evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        best_score = -1
        best_model_name = None
        best_metrics = None
        
        # Train all models if not already trained
        self.train_all_models(X_train, y_train)
        
        for model_name in self.models.keys():
            if use_cross_validation:
                cv_results = self.cross_validate_model(model_name, X_train, y_train)
                current_score = cv_results['mean_score']
            else:
                # Use test set performance
                y_pred = self.predict(model_name, X_test)
                metrics = evaluator.evaluate_model(y_test, y_pred)
                current_score = metrics['accuracy']
            
            if current_score > best_score:
                best_score = current_score
                best_model_name = model_name
                if not use_cross_validation:
                    best_metrics = metrics
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        if use_cross_validation:
            # Get final metrics on test set
            y_pred = self.predict(best_model_name, X_test)
            best_metrics = evaluator.evaluate_model(y_test, y_pred)
        
        return best_model_name, best_metrics
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to file.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available.")
        
        joblib.dump(self.models[model_name], filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to load the model from
        """
        self.models[model_name] = joblib.load(filepath)
        print(f"Model loaded as {model_name} from {filepath}")
    
    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            directory (str): Directory to save models
        """
        import os
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for model_name in self.models.keys():
            filepath = os.path.join(directory, f"{model_name}.joblib")
            self.save_model(model_name, filepath)
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all models.
        
        Returns:
            Dict: Model information
        """
        model_info = {}
        
        for model_name, model in self.models.items():
            info = {
                'type': type(model).__name__,
                'parameters': model.get_params(),
                'trained': hasattr(model, 'classes_') if hasattr(model, 'classes_') else False
            }
            model_info[model_name] = info
        
        return model_info
    
    def predict_with_best_model(self, X_test) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X_test: Test features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run find_best_model first.")
        
        return self.best_model.predict(X_test)
