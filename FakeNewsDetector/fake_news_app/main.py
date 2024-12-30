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

    # 1. Vectorize the input text
    vectorized_text = tfidf_vectorizer.transform([input_text])

    # 2. Predict the label
    prediction = pac.predict(vectorized_text)[0]

    # 3. (Optional) Confidence-like measure (distance from decision boundary)
    try:
        distance = pac.decision_function(vectorized_text)[0]
        # Sign indicates FAKE vs REAL
        # Magnitude is how far from the boundary
        confidence = float(abs(distance))
    except:
        confidence = None

    # 4. Return a JSON response
    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    # For local development
    app.run(debug=True)
