from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
model = joblib.load('restaurant_review_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask application
app = Flask(__name__)

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['processed_review']  # Get the review text from the form

        # Preprocess the review text using the vectorizer
        review_vectorized = vectorizer.transform([review])

        # Make prediction
        predicted_rating = model.predict(review_vectorized)[0]

        if predicted_rating in [1, 2]:
            sentiment = "Negatif"
        elif predicted_rating == 3:
            sentiment = "Netral"
        elif predicted_rating in [4, 5]:
            sentiment = "Positif"
        else:
            sentiment = "Tidak Diketahui"
        # Return the result in a template
        return render_template('result.html', review=review, predicted_rating=predicted_rating, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
