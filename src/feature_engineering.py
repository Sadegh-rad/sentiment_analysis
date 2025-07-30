"""
Feature Engineering Module

This module contains functions and classes for extracting features from 
preprocessed text data for sentiment analysis models.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt


class FeatureExtractor:
    """
    Feature extraction class for sentiment analysis.
    
    This class provides methods for extracting various types of features
    from text data including TF-IDF vectors and count vectors.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature extractor.
        
        Args:
            random_state (int): Random state for reproducible results
        """
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.feature_names = None
    
    def setup_tfidf_vectorizer(self, min_df: int = 5, max_df: float = 0.8,
                              sublinear_tf: bool = True, use_idf: bool = True,
                              max_features: Optional[int] = None) -> None:
        """
        Setup TF-IDF vectorizer with specified parameters.
        
        Args:
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            sublinear_tf (bool): Whether to use sublinear TF scaling
            use_idf (bool): Whether to use inverse document frequency
            max_features (int): Maximum number of features
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            use_idf=use_idf,
            max_features=max_features
        )
    
    def setup_count_vectorizer(self, min_df: int = 1, max_df: float = 1.0,
                              max_features: Optional[int] = None) -> None:
        """
        Setup Count vectorizer with specified parameters.
        
        Args:
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            max_features (int): Maximum number of features
        """
        self.count_vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )
    
    def fit_tfidf(self, texts: pd.Series) -> None:
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts (pd.Series): Training texts
        """
        if self.tfidf_vectorizer is None:
            self.setup_tfidf_vectorizer()
        
        self.tfidf_vectorizer.fit(texts.astype(str))
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f'Number of TF-IDF features: {len(self.feature_names)}')
    
    def transform_tfidf(self, texts: pd.Series):
        """
        Transform texts using fitted TF-IDF vectorizer.
        
        Args:
            texts (pd.Series): Texts to transform
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        return self.tfidf_vectorizer.transform(texts.astype(str))
    
    def fit_transform_tfidf(self, texts: pd.Series):
        """
        Fit TF-IDF vectorizer and transform texts in one step.
        
        Args:
            texts (pd.Series): Texts to fit and transform
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            self.setup_tfidf_vectorizer()
        
        features = self.tfidf_vectorizer.fit_transform(texts.astype(str))
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f'Number of TF-IDF features: {len(self.feature_names)}')
        
        return features
    
    def analyze_word_frequency(self, texts: pd.Series, top_n: int = 20,
                              figsize: Tuple[int, int] = (15, 10)) -> pd.DataFrame:
        """
        Analyze and visualize word frequency in the corpus.
        
        Args:
            texts (pd.Series): Text data
            top_n (int): Number of top words to show
            figsize (tuple): Figure size for plot
            
        Returns:
            pd.DataFrame: Word frequency dataframe
        """
        # Setup count vectorizer if not exists
        if self.count_vectorizer is None:
            self.setup_count_vectorizer()
        
        # Fit and transform texts
        word_counts = self.count_vectorizer.fit_transform(texts.astype(str))
        
        # Get word frequencies
        sum_words = word_counts.sum(axis=0)
        words_freq = [
            (word, sum_words[0, idx]) 
            for word, idx in self.count_vectorizer.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        # Create dataframe
        frequency_df = pd.DataFrame(
            words_freq[:top_n], 
            columns=['words', 'frequency']
        )
        
        # Create visualization
        plt.figure(figsize=figsize)
        frequency_df.plot(
            x='words', 
            y='frequency', 
            kind='barh', 
            figsize=figsize, 
            color='steelblue'
        )
        plt.gca().invert_yaxis()
        plt.ylabel('Words', fontsize=14)
        plt.xlabel('Frequency', fontsize=14)
        plt.title(f'Top {top_n} Most Frequent Words', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return frequency_df
    
    def split_data(self, X: pd.Series, y: pd.Series, test_size: float = 0.2,
                  stratify: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.Series): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            stratify (bool): Whether to stratify split
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify else None
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
    
    def prepare_features(self, train_texts: pd.Series, test_texts: pd.Series,
                        eval_texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Prepare all features for training and testing.
        
        Args:
            train_texts (pd.Series): Training texts
            test_texts (pd.Series): Testing texts
            eval_texts (pd.Series, optional): Evaluation texts
            
        Returns:
            Dict: Dictionary containing all feature matrices
        """
        # Fit TF-IDF on training data
        self.fit_tfidf(train_texts)
        
        # Transform all datasets
        X_train = self.transform_tfidf(train_texts)
        X_test = self.transform_tfidf(test_texts)
        
        features = {
            'X_train': X_train,
            'X_test': X_test,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names)
        }
        
        if eval_texts is not None:
            X_eval = self.transform_tfidf(eval_texts)
            features['X_eval'] = X_eval
        
        return features
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained sklearn model with feature_importances_ or coef_ attribute
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_names is None:
            raise ValueError("No features available. Run feature extraction first.")
        
        # Get importance scores
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0])
        else:
            raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        ).head(top_n)
        
        return feature_importance
    
    def visualize_feature_importance(self, model, top_n: int = 20,
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize feature importance.
        
        Args:
            model: Trained sklearn model
            top_n (int): Number of top features to show
            figsize (tuple): Figure size
        """
        feature_importance = self.get_feature_importance(model, top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def save_vectorizer(self, filepath: str, vectorizer_type: str = 'tfidf') -> None:
        """
        Save fitted vectorizer to file.
        
        Args:
            filepath (str): Path to save the vectorizer
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
        """
        import joblib
        
        if vectorizer_type == 'tfidf' and self.tfidf_vectorizer is not None:
            joblib.dump(self.tfidf_vectorizer, filepath)
        elif vectorizer_type == 'count' and self.count_vectorizer is not None:
            joblib.dump(self.count_vectorizer, filepath)
        else:
            raise ValueError(f"No fitted {vectorizer_type} vectorizer found")
    
    def load_vectorizer(self, filepath: str, vectorizer_type: str = 'tfidf') -> None:
        """
        Load vectorizer from file.
        
        Args:
            filepath (str): Path to load the vectorizer from
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
        """
        import joblib
        
        if vectorizer_type == 'tfidf':
            self.tfidf_vectorizer = joblib.load(filepath)
            if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
                self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        elif vectorizer_type == 'count':
            self.count_vectorizer = joblib.load(filepath)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
