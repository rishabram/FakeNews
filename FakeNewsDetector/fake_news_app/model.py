import re
import string
import nltk
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# If you haven't downloaded these before, uncomment and run once.
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------------------------------------------
# 1. Data Cleaning Function
# -----------------------------------------------------------------------------
def clean_text(text):
    """Basic text-cleaning function:
       - Lowercase
       - Remove URLs, punctuation, digits
       - Tokenize, remove stopwords
       - Lemmatize
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = nltk.word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    # Re-join
    cleaned_text = ' '.join(words)
    return cleaned_text

# -----------------------------------------------------------------------------
# 2. Training & Tuning the Model
# -----------------------------------------------------------------------------
def train_fake_news_model(csv_path='news.csv'):
    """Train a PassiveAggressiveClassifier with TF-IDF vectorizer,
       using data cleaning + GridSearchCV for better performance.
    """

    # 2.1 Load dataset
    df = pd.read_csv(csv_path)
    print("Dataset shape:", df.shape)

    # Optional preview
    print(df.head())

    # 2.2 Preprocess the text
    #    Apply cleaning to each row in 'text' column
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 2.3 Extract features (cleaned text) and labels
    X = df['cleaned_text']
    y = df['label']

    # 2.4 Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2.5 Build a Pipeline for TF-IDF + PassiveAggressiveClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('pac', PassiveAggressiveClassifier(random_state=42))
    ])

    # 2.6 Define hyperparameters to tune
    param_grid = {
        'tfidf__stop_words': [None, 'english'],
        'tfidf__max_df': [0.7, 1.0],
        'tfidf__ngram_range': [(1,1), (1,2)],  # unigrams or bigrams
        'pac__C': [0.01, 0.1, 1.0, 10],
        'pac__max_iter': [50, 100, 200],      # more iterations for possible better convergence
        'pac__early_stopping': [True],        # allow early stopping
    }

    # 2.7 Hyperparameter tuning with GridSearchCV
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,              # 5-fold cross-validation
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    # 2.8 Fit GridSearch on training data
    print("Starting Grid Search for best parameters...")
    grid_search.fit(X_train, y_train)

    print("Best Params:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # 2.9 Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {round(test_accuracy*100, 2)}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print("Confusion Matrix:\n", cm)

    # 2.10 Return best model + vectorizer
    #      We'll separate them for the Flask app
    #      i.e. best_model has a .named_steps['tfidf'] and .named_steps['pac']
    tfidf_vectorizer = best_model.named_steps['tfidf']
    pac_model = best_model.named_steps['pac']

    return tfidf_vectorizer, pac_model
