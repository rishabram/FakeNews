from flask import Flask, request, jsonify
from model import train_fake_news_model

app = Flask(__name__)

# Train or load your model at startup
tfidf_vectorizer, pac = train_fake_news_model('news.csv')

@app.route('/')
def home():
    return "Fake News Detector is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON body with a field 'text' containing
    the text to classify, e.g. { "text": "Some suspicious news text" }
    """
    data = request.get_json()
    input_text = data.get('text', '')

    # Vectorize the input text
    vectorized_text = tfidf_vectorizer.transform([input_text])

    # Predict the label
    prediction = pac.predict(vectorized_text)[0]

    # If you want a confidence-like measure, you can use decision_function
    # But remember, PassiveAggressiveClassifier doesn't output probabilities
    confidence = None
    try:
        distance = pac.decision_function(vectorized_text)[0]
        # The sign of distance determines FAKE vs REAL
        # The magnitude is how far from the decision boundary
        confidence = float(abs(distance))
    except:
        pass


    # Return a JSON response
    return jsonify({
        'prediction': prediction,
        'confidence': confidence  # or remove if not needed
    })

if __name__ == '__main__':
    app.run(debug=True)