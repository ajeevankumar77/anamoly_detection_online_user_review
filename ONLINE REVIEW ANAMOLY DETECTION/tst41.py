import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Function to display anomaly reviews and their clean text
def display_anomaly_reviews(df, anomaly_indices):
    anomaly_reviews = []
    for idx in anomaly_indices:
        anomaly_reviews.append({
            'anomaly_review': df.loc[idx, 'review_text'],
            'clean_text': df.loc[idx, 'clean_text']
        })
    return anomaly_reviews

# Read the CSV file and perform anomaly detection
def anomaly_detection_on_reviews(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Text preprocessing
    df['clean_text'] = df['review_text'].apply(preprocess_text)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])

    # Anomaly detection using Isolation Forest
    isolation_forest = IsolationForest(contamination=0.1)  # Adjust contamination based on expected anomaly rate
    isolation_forest.fit(X)

    # Predict anomalies
    anomaly_predictions = isolation_forest.predict(X)

    # Get the indices of anomaly reviews
    anomaly_indices = [i for i, prediction in enumerate(anomaly_predictions) if prediction == -1]

    # Display anomaly reviews and their clean text
    anomaly_reviews = display_anomaly_reviews(df, anomaly_indices)

    return anomaly_reviews

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = 'uploads/customer_reviews.csv'
    file.save(file_path)  # Save the uploaded file
    anomaly_reviews = anomaly_detection_on_reviews(file_path)
    return jsonify({'anomaly_reviews': anomaly_reviews})

if __name__ == '__main__':
    app.run(debug=True)
