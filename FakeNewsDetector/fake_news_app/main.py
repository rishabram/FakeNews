from flask import Flask, request, jsonify
from flask_cors import CORS

from FakeNewsDetector.fake_news_app.model import train_fake_news_model

app = Flask(__name__)
CORS(app)  # enables cross-origin requests

# Train or load your model at startup
tfidf_vectorizer, pac = train_fake_news_model('news.csv')


@app.route('/')
def home():
    return "Fake News Detector is running!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')
    vectorized_text = tfidf_vectorizer.transform([input_text])
    prediction = pac.predict(vectorized_text)[0]

    confidence = None
    try:
        distance = pac.decision_function(vectorized_text)[0]
        confidence = float(abs(distance))
    except:
        pass

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
