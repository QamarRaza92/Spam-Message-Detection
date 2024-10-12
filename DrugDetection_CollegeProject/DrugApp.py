from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re
import string

# Load the trained model and vectorizer
model = joblib.load('drug_detection_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the clean_text function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespaces
    text = text.strip()

    return text

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('predict_form.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from request
    post_text = request.form.get('post_text')
    ip_address = request.form.get('ip_address')

    # Check if post_text or ip_address are missing
    if not post_text or not ip_address:
        return jsonify({'error': 'Both post_text and ip_address are required'}), 400

    # Clean the post text
    text_cleaned = clean_text(post_text)

    # Vectorize the cleaned post text
    vectorized_text = vectorizer.transform([text_cleaned])

    # Create features array (vectorized post text + IP length)
    features = pd.concat([pd.DataFrame(vectorized_text.toarray()), pd.DataFrame([[len(ip_address)]])], axis=1)

    # Make the prediction using the loaded model
    prediction = model.predict(features)
    if int(prediction[0])==1:
        prediction = 'User Suspected'
    else:
        prediction = 'User Suspected'

    # Return the result by re-rendering the form with the prediction
    return render_template('predict_form.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
