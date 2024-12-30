import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_fake_news_model(csv_path='news.csv'):
    # 1. Load dataset
    df = pd.read_csv(csv_path)

    # 2. Examine the DataFrame (optional)
    print("Dataset shape:", df.shape)
    print(df.head())

    # 3. Extract features (text) and labels
    X = df['text']
    y = df['label']

    # 4. Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # 5. Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # 6. Fit and transform the training set, transform the test set
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # 7. Initialize PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # 8. Evaluate on the test set
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100, 2)}%')

    # 9. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
    print("Confusion Matrix:\n", cm)

    # 10. Return the fitted vectorizer and model
    return tfidf_vectorizer, pac