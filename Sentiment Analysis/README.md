# Sentiment Analysis Project

A sentiment analysis project that classifies tweet sentiments using machine learning models (SVM, Logistic Regression, Naive Bayes).

## Features

- Text preprocessing (cleaning, tokenization, lemmatization)
- TF-IDF feature extraction
- Multiple ML models with comparison
- Data balancing and visualization

## Dataset

- `development.csv`: Training data (224,996 samples)
- `evaluation.csv`: Test data (75,001 samples)
- Binary classification: 0 (negative), 1 (positive)

## Project Structure

```
sentiment_analysis/
├── README.md
├── requirements.txt
├── data/raw/              # Place your CSV files here
├── src/                   # Source code modules
├── notebooks/             # Jupyter analysis
├── results/               # Output predictions
└── docs/                  # Original report
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sadegh-rad/sentiment_analysis.git
cd sentiment_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data files in `data/raw/`:
   - `development.csv`
   - `evaluation.csv`

## Usage

### Quick Run
```bash
python src/main.py
```

### Interactive Analysis
```bash
jupyter notebook notebooks/sentiment_analysis_complete.ipynb
```

### Using Individual Modules
```python
from src.data_preprocessing import SentimentDataProcessor
from src.models import SentimentClassifier

# Preprocess data
processor = SentimentDataProcessor()
data = processor.prepare_data('data/raw/development.csv', 'data/raw/evaluation.csv')

# Train models
classifier = SentimentClassifier()
classifier.train_all_models(X_train, y_train)

# Make predictions
predictions = classifier.predict('logistic_regression', X_test)
```

## Results

Expected model performance:
- **Logistic Regression**: ~81% accuracy
- **Linear SVM**: ~80% accuracy  
- **Naive Bayes**: ~77% accuracy

Results are saved in:
- `results/final_predictions.csv` - Model predictions
- `results/evaluation_results.csv` - Performance metrics

## Text Preprocessing Steps

1. Lowercase conversion
2. Remove URLs, usernames, HTML tags
3. Expand contractions (won't → will not)
4. Remove numbers and special characters
5. Remove short words and stopwords
6. Tokenization and lemmatization

---

**Note**: This project preserves the original methodology from the Data Science Lab assignment while providing a clean, modular implementation.
