"""
Data Preprocessing Module

This module contains functions and classes for preprocessing social media text data
for sentiment analysis. It includes comprehensive text cleaning, normalization,
and preparation functions.
"""

import pandas as pd
import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.utils import resample
from typing import List, Dict, Union, Optional


class SentimentDataProcessor:
    """
    A comprehensive data preprocessing class for sentiment analysis.
    
    This class provides methods for loading, cleaning, and preprocessing
    text data for machine learning models.
    """
    
    def __init__(self, random_state: int = 10):
        """
        Initialize the data processor.
        
        Args:
            random_state (int): Random state for reproducible results
        """
        self.random_state = random_state
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
        
        # Define stopwords list
        self.stopwords = {
            'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
            'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
            'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
            'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
            'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
            'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
            'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
            'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
            'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes",
            'should', "shouldve", 'so', 'some', 'such',
            't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
            'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
            'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
            'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
            'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
            "youve", 'your', 'yours', 'yourself', 'yourselves', 'aww', 'loud', 'get', 'quot', 'amp', 'would',
            'could', 'yes', 'though', 'but', 'haha', 'hahaha', 'dont', 'cant', 'even', 'tho', 'already',
            'yet', 'hehe', 'lot', 'love', 'think', 'know', 'one', 'go', 'today', 'see', 'time', 'work', 'make', 'say', 'yeah', 'way', 'laugh'
        }
    
    def load_data(self, filepath: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            encoding (str): File encoding
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath, sep=',', encoding=encoding, index_col=False)
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {str(e)}")
    
    def balance_data(self, df: pd.DataFrame, target_col: str = 'sentiment', 
                    minority_class: int = 0) -> pd.DataFrame:
        """
        Balance the dataset by downsampling the majority class.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            minority_class (int): Label of the minority class
            
        Returns:
            pd.DataFrame: Balanced dataframe
        """
        # Get class counts
        class_counts = df[target_col].value_counts()
        minority_count = class_counts[minority_class]
        
        # Separate classes
        df_majority = df[df[target_col] != minority_class]
        df_minority = df[df[target_col] == minority_class]
        
        # Downsample majority class
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=minority_count,
            random_state=self.random_state
        )
        
        # Combine classes
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
        return df_balanced.reset_index(drop=True)
    
    def clean_usernames(self, text: str) -> str:
        """Remove @mentions from text."""
        return re.sub('@[^\\s]+', '', text)
    
    def clean_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\\S+|www\\.\\S+')
        return url_pattern.sub('', text)
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub('', text)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with expanded contractions
        """
        contractions = {
            r"won\\'t": " will not",
            r"won\\'t\\'ve": " will not have",
            r"can\\'t": " can not",
            r"don\\'t": " do not",
            r"can\\'t\\'ve": " can not have",
            r"ma\\'am": " madam",
            r"let\\'s": " let us",
            r"ain\\'t": " am not",
            r"shan\\'t": " shall not",
            r"sha\\n\\'t": " shall not",
            r"o\\'clock": " of the clock",
            r"y\\'all": " you all",
            r"n\\'t": " not",
            r"n\\'t\\'ve": " not have",
            r"\\'re": " are",
            r"\\'s": " is",
            r"\\'d": " would",
            r"\\'d\\'ve": " would have",
            r"\\'ll": " will",
            r"\\'ll\\'ve": " will have",
            r"\\'t": " not",
            r"\\'ve": " have",
            r"\\'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        return text
    
    def separate_alphanumeric(self, text: str) -> str:
        """Separate alphanumeric characters."""
        words = re.findall(r"[^\\W\\d_]+|\\d+", text)
        return " ".join(words)
    
    def remove_numbers(self, text: str) -> str:
        """Remove numeric characters."""
        return re.sub('[0-9]+', '', text)
    
    def keep_only_letters(self, text: str) -> str:
        """Keep only alphabetic characters."""
        return re.sub(r'[^a-zA-Z]', ' ', text)
    
    def remove_short_words(self, text: str, min_length: int = 3) -> str:
        """Remove words shorter than specified length."""
        words = text.split()
        return ' '.join([word for word in words if len(word) >= min_length])
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        return " ".join([word for word in words if word.lower() not in self.stopwords])
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        translator = str.maketrans('', '', string.punctuation)
        return str(text).translate(translator)
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return self.tokenizer.tokenize(text)
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens based on their POS tags.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        lemmatized_tokens = []
        
        for word, tag in pos_tag(tokens):
            # Convert POS tag to format expected by WordNetLemmatizer
            if tag.startswith('NN'):
                pos = 'n'  # noun
            elif tag.startswith('VB'):
                pos = 'v'  # verb
            else:
                pos = 'a'  # adjective
            
            lemmatized_tokens.append(self.lemmatizer.lemmatize(word, pos))
        
        return lemmatized_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply complete text preprocessing pipeline.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Apply cleaning functions
        text = self.clean_usernames(text)
        text = self.clean_urls(text)
        text = self.clean_html(text)
        text = self.expand_contractions(text)
        text = self.separate_alphanumeric(text)
        text = self.remove_numbers(text)
        text = self.keep_only_letters(text)
        text = self.remove_short_words(text)
        text = self.remove_stopwords(text)
        text = self.remove_punctuation(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_text(text)
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to string
        return " ".join(lemmatized_tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str = 'text', 
                           target_col: str = 'sentiment', 
                           balance_data: bool = True) -> pd.DataFrame:
        """
        Preprocess entire dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_col (str): Name of text column
            target_col (str): Name of target column
            balance_data (bool): Whether to balance the data
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Balance data if requested and target column exists
        if balance_data and target_col in processed_df.columns:
            processed_df = self.balance_data(processed_df, target_col)
        
        # Preprocess text
        processed_df[text_col] = processed_df[text_col].apply(self.preprocess_text)
        
        return processed_df
    
    def prepare_data(self, train_path: str, test_path: str, 
                    text_col: str = 'text', target_col: str = 'sentiment') -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess both training and test data.
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
            text_col (str): Name of text column
            target_col (str): Name of target column
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing processed train and test data
        """
        # Load data
        train_df = self.load_data(train_path)
        test_df = self.load_data(test_path)
        
        # Preprocess training data (with balancing)
        train_processed = self.preprocess_dataframe(
            train_df, text_col, target_col, balance_data=True
        )
        
        # Preprocess test data (without balancing)
        test_processed = self.preprocess_dataframe(
            test_df, text_col, target_col, balance_data=False
        )
        
        return {
            'train': train_processed,
            'test': test_processed
        }
