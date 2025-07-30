"""
Sentiment Analysis Package

A comprehensive sentiment analysis toolkit for social media text classification.
"""

__version__ = "1.0.0"
__author__ = "Sadegh Rad"
__email__ = "sadegh.rad@example.com"

# Import classes when available
try:
    from .data_preprocessing import SentimentDataProcessor
    from .feature_engineering import FeatureExtractor
    from .models import SentimentClassifier
    from .evaluation import ModelEvaluator
    
    __all__ = [
        "SentimentDataProcessor",
        "FeatureExtractor", 
        "SentimentClassifier",
        "ModelEvaluator"
    ]
except ImportError:
    # Handle cases where dependencies might not be installed
    __all__ = []
