from flask import Flask, request, jsonify
from model import train_fake_news_model

app = Flask(__name__)

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

    if confidence is not None:
        if confidence > 1.5:
            confidence_label = "Most Likely " + ("True" if prediction == "REAL" else "False")
        elif confidence > 0.5:
            confidence_label = "Maybe " + ("True" if prediction == "REAL" else "False")
        else:
            confidence_label = prediction+ " but verify with other sources!"
    else:
        confidence_label = "No confidence available"

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'confidence_label': confidence_label
    })


if __name__ == '__main__':
    app.run(debug=True)