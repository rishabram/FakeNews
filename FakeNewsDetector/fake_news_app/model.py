import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import string

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def train_fake_news_model(csv_path='news.csv'):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    # 1. Load dataset
    df = pd.read_csv(csv_path)
    print("Dataset shape:", df.shape)
    print(df.head())

    # 2. Clean the text
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 3. Extract features (text) and labels
    X = df['cleaned_text']
    y = df['label']

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # 5. TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # 6. PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # 7. Evaluate
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100, 2)}%')

    cm = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
    print("Confusion Matrix:\n", cm)

    return tfidf_vectorizer, pac
