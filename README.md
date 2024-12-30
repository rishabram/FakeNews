# Fake News Detector

**Table of Contents**

1. [Overview](#overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Project Structure](#project-structure)  
5. [Model Training & Inference](#model-training--inference)  
6. [Flask Backend Setup](#flask-backend-setup)  
7. [Chrome Extension Setup](#chrome-extension-setup)  
8. [Deployment](#deployment)  
9. [Future Improvements](#future-improvements)  
10. [References](#references)

---

## Overview

**Fake News Detector** is a project that identifies potentially **fake** or **real** news content using a **Machine Learning (ML)** model. It comes with a **Chrome Extension** that lets users highlight any text on a webpage and quickly get a classification result (**FAKE** vs. **REAL**), along with a confidence/distance score.

This project demonstrates:
1. **Natural Language Processing (NLP)** with TF-IDF.
2. **Machine Learning** model building (scikit-learn).
3. **Flask** for serving a RESTful prediction API.
4. **Chrome Extension** development (Manifest V3).
5. **Deployment** considerations (e.g., Heroku).

---

## Features

- **Train** an ML model on a labeled dataset of fake and real news.
- **Classify** user-highlighted text on any webpage in real-time using a Chrome Extension.
- **Confidence or Distance Score** (depending on classifier capabilities).
- **User-Friendly** interface via the extension’s popup.

---

## Tech Stack

- **Python 3.7+**  
- **Flask** (for the backend)  
- **scikit-learn**, **pandas**, **numpy** (for ML)  
- **Chrome Extension** (Manifest V3)  
- **Heroku** (optional deployment)

---

## Project Structure


```plaintext
FakeNewsDetector/
├── fake_news_app/
│   ├── main.py         # Flask app entry point
│   ├── model.py        # Model training logic
│   ├── news.csv        # Dataset
│   └── requirements.txt # Dependencies
└── chrome_extension/
    ├── manifest.json    # Chrome extension descriptor (Manifest V3)
    ├── popup.html       # Popup interface
    ├── popup.js         # Chrome extension logic
    ├── background.js    # (Optional) service worker or additional extension logic
```

### In `fake_news_app/`
- **`model.py`**  
  Contains the logic to train and evaluate the model using `TfidfVectorizer` and `PassiveAggressiveClassifier`.  
- **`main.py`**  
  A Flask app that loads or trains the model on startup, then exposes a `/predict` endpoint for inference.  
- **`news.csv`**  
  A sample dataset for fake/real news classification.  
- **`requirements.txt`**  
  Lists Python dependencies (Flask, scikit-learn, gunicorn, pandas, etc.).

### In `chrome_extension/`
- **`manifest.json`**  
  Chrome Extension configuration (permissions, name, version, etc.).  
- **`popup.html`**  
  The HTML UI displayed when clicking the extension’s icon.  
- **`popup.js`**  
  Handles user interactions (e.g. capturing highlighted text, calling Flask API).  
- **`background.js`**  
  An optional service worker file for background tasks.

---

## Model Training & Inference

1. **Training**  
   - We typically use `model.py` to train the model on `news.csv`.  
   - Core steps:
     1. **Load** the dataset into a Pandas DataFrame.  
     2. **Split** into training and test sets (e.g., 80/20).  
     3. **Vectorize** the text with `TfidfVectorizer`.  
     4. **Train** a `PassiveAggressiveClassifier` (or alternative).  
     5. **Evaluate** accuracy and confusion matrix.  

2. **Inference**  
   - The Flask app (`main.py`) loads the trained vectorizer and model at startup.  
   - `/predict` endpoint receives JSON input (`text`), vectorizes it, and returns a JSON response (FAKE or REAL + confidence/distance measure).

---

## Flask Backend Setup

1. **Install Requirements**
   ```bash
   cd fake_news_app
   pip install -r requirements.txt

2. **Train the Model (Optional if main.py automatically trains on startup, but you can also pre-train and save the model)**

```bash
python model.py
```
Adjust your scripts accordingly if you’re saving/loading a model pickle.

3. **Run the Flask App**

```bash
python main.py
```
By default, the server starts at: http://127.0.0.1:5000.
Test the Endpoint

Use curl or a tool like Postman:
bash
Copy code
curl -X POST -H "Content-Type: application/json" \
-d '{"text": "Suspicious news text"}' \
http://127.0.0.1:5000/predict
You should receive a JSON response with prediction and confidence.
Chrome Extension Setup
Load the Extension into Chrome

Go to chrome://extensions/.
Enable Developer mode (toggle on top-right).
Click Load unpacked.
Select the chrome_extension/ folder.
Configure the Endpoint

In popup.js, find the line where it calls the Flask endpoint:
js
Copy code
const flaskUrl = "http://127.0.0.1:5000/predict";
If you deploy your Flask app (e.g. on Heroku), update this to your production URL.
Usage

Navigate to any webpage.
Highlight some text you want to check.
Click on the Fake News Detector extension icon.
Hit the “Check Selected Text” button.
The popup will show whether the text is likely FAKE or REAL (and optionally a confidence score).
Deployment
Deploying to Heroku
Make sure you have a Procfile in fake_news_app/:
makefile
Copy code
web: gunicorn main:app
Ensure your requirements.txt includes gunicorn.
Create and push to Heroku:
bash
Copy code
heroku create your-fake-news-app
git init
git add .
git commit -m "Initial commit"
heroku git:remote -a your-fake-news-app
git push heroku master
Once deployed, Heroku gives you a URL (e.g., https://your-fake-news-app.herokuapp.com).
In your Chrome Extension (popup.js), update flaskUrl to:
js
Copy code
const flaskUrl = "https://your-fake-news-app.herokuapp.com/predict";
Future Improvements
Probability Scores

Consider using LogisticRegression or SGDClassifier(loss='log') to get probability outputs (predict_proba).
Advanced NLP

Integrate data-cleaning, lemmatization, or advanced embeddings (e.g., Word2Vec, BERT).
Frontend Enhancement

Provide a more polished UI in the Chrome Extension, or highlight the text in-page with color-coded results.
Caching & Scalability

For higher traffic, consider caching results or scaling the Flask app with a production-grade server and database.
References
Detecting Fake News with Python and Machine Learning - DataFlair
How to Build a Fake News Classification Model (ODS)
Chrome Extension Official Docs
Flask Documentation
Heroku Deployment Guide
Author: [Your Name or Team Name]
LinkedIn | Portfolio | Twitter

Feel free to star this repository if you found it helpful! If you have any questions, please open an issue or reach out.

License: Choose an appropriate license, e.g. MIT License.

Thank you for checking out the Fake News Detector!

<div align="center">

</div>
