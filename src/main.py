
"""
Main execution script for sentiment analysis pipeline.

This script demonstrates how to use the sentiment analysis package
to preprocess data, extract features, train models, and evaluate results.
"""

import os
import pandas as pd
import numpy as np

# Add src to path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import SentimentDataProcessor
from feature_engineering import FeatureExtractor
from models import SentimentClassifier
from evaluation import ModelEvaluator


def main():
    """Main function to run the sentiment analysis pipeline."""
    
    print("="*60)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    processor = SentimentDataProcessor(random_state=42)
    feature_extractor = FeatureExtractor(random_state=42)
    classifier = SentimentClassifier(random_state=42)
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    print("\n2. Loading and preprocessing data...")
    try:
        # Load training data
        train_df = processor.load_data('data/raw/development.csv')
        print(f"Loaded training data: {train_df.shape}")
        
        # Load test data
        test_df = processor.load_data('data/raw/evaluation.csv')
        print(f"Loaded test data: {test_df.shape}")
        
        # Preprocess training data
        train_processed = processor.preprocess_dataframe(
            train_df, 
            text_col='text', 
            target_col='sentiment', 
            balance_data=True
        )
        print(f"Processed training data: {train_processed.shape}")
        
        # Preprocess test data
        test_processed = processor.preprocess_dataframe(
            test_df, 
            text_col='text', 
            balance_data=False
        )
        print(f"Processed test data: {test_processed.shape}")
        
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return
    
    # Split data for training and validation
    print("\n3. Splitting data...")
    x_data = train_processed['text'].astype(str)
    y_data = train_processed['sentiment'].astype(str)
    
    x_train, x_val, y_train, y_val = feature_extractor.split_data(
        x_data, y_data, test_size=0.2
    )
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    
    # Feature extraction
    print("\n4. Extracting features...")
    x_test_eval = test_processed['text'].astype(str)
    features = feature_extractor.prepare_features(x_train, x_val, x_test_eval)
    
    x_train_tfidf = features['X_train']
    x_val_tfidf = features['X_test']  # This is validation set
    x_eval_tfidf = features['X_eval']  # This is final test set
    
    print(f"TF-IDF features: {features['n_features']} dimensions")
    
    # Train models
    print("\n5. Training models...")
    classifier.train_all_models(x_train_tfidf, y_train)
    
    # Evaluate models on validation set
    print("\n6. Evaluating models...")
    models_to_test = ['linear_svm', 'logistic_regression', 'naive_bayes']
    
    for model_name in models_to_test:
        print(f"\nEvaluating {model_name}:")
        y_pred = classifier.predict(model_name, x_val_tfidf)
        evaluator.evaluate_model(y_val, y_pred, model_name)
        evaluator.print_classification_report(y_val, y_pred, model_name)
    
    # Find best model
    print("\n7. Finding best model...")
    best_model_name, best_metrics = classifier.find_best_model(
        x_train_tfidf, y_train, x_val_tfidf, y_val
    )
    
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
    
    # Make final predictions
    print("\n8. Making final predictions...")
    final_predictions = classifier.predict_with_best_model(x_eval_tfidf)
    
    # Save predictions
    os.makedirs('results', exist_ok=True)
    predictions_df = pd.DataFrame(final_predictions, columns=['Predicted'])
    predictions_df.index.name = 'Id'
    predictions_df.to_csv('results/final_predictions.csv')
    
    # Save evaluation results
    evaluator.save_evaluation_results('results/evaluation_results.csv')
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
    print("Results saved to 'results/' directory")
    print("="*60)


if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import SentimentDataProcessor
from feature_engineering import FeatureExtractor
from models import SentimentClassifier
from evaluation import ModelEvaluator


def main():
    """Main function to run the sentiment analysis pipeline."""
    
    print("="*60)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    processor = SentimentDataProcessor(random_state=42)
    feature_extractor = FeatureExtractor(random_state=42)
    classifier = SentimentClassifier(random_state=42)
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    print("\n2. Loading and preprocessing data...")
    try:
        # Load training data
        train_df = processor.load_data('data/raw/development.csv')
        print(f"Loaded training data: {train_df.shape}")
        
        # Load test data
        test_df = processor.load_data('data/raw/evaluation.csv')
        print(f"Loaded test data: {test_df.shape}")
        
        # Preprocess training data
        train_processed = processor.preprocess_dataframe(
            train_df, 
            text_col='text', 
            target_col='sentiment', 
            balance_data=True
        )
        print(f"Processed training data: {train_processed.shape}")
        
        # Preprocess test data
        test_processed = processor.preprocess_dataframe(
            test_df, 
            text_col='text', 
            balance_data=False
        )
        print(f"Processed test data: {test_processed.shape}")
        
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return
    
    # Split data for training and validation
    print("\n3. Splitting data...")
    x_data = train_processed['text'].astype(str)
    y_data = train_processed['sentiment'].astype(str)
    
    x_train, x_val, y_train, y_val = feature_extractor.split_data(
        x_data, y_data, test_size=0.2
    )
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    
    # Feature extraction
    print("\n4. Extracting features...")
    x_test_eval = test_processed['text'].astype(str)
    features = feature_extractor.prepare_features(x_train, x_val, x_test_eval)
    
    x_train_tfidf = features['X_train']
    x_val_tfidf = features['X_test']  # This is validation set
    x_eval_tfidf = features['X_eval']  # This is final test set
    
    print(f"TF-IDF features: {features['n_features']} dimensions")
    
    # Train models
    print("\n5. Training models...")
    classifier.train_all_models(x_train_tfidf, y_train)
    
    # Evaluate models on validation set
    print("\n6. Evaluating models...")
    models_to_test = ['linear_svm', 'logistic_regression', 'naive_bayes']
    
    for model_name in models_to_test:
        print(f"\nEvaluating {model_name}:")
        y_pred = classifier.predict(model_name, x_val_tfidf)
        evaluator.evaluate_model(y_val, y_pred, model_name)
        evaluator.print_classification_report(y_val, y_pred, model_name)
    
    # Find best model
    print("\n7. Finding best model...")
    best_model_name, best_metrics = classifier.find_best_model(
        x_train_tfidf, y_train, x_val_tfidf, y_val
    )
    
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
    
    # Make final predictions
    print("\n8. Making final predictions...")
    final_predictions = classifier.predict_with_best_model(x_eval_tfidf)
    
    # Save predictions
    os.makedirs('results', exist_ok=True)
    predictions_df = pd.DataFrame(final_predictions, columns=['Predicted'])
    predictions_df.index.name = 'Id'
    predictions_df.to_csv('results/final_predictions.csv')
    
    # Save evaluation results
    evaluator.save_evaluation_results('results/evaluation_results.csv')
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
    print("Results saved to 'results/' directory")
    print("="*60)


if __name__ == "__main__":
    main()
