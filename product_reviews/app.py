from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# NLP setup
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Store history
history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = clean_text(review)
        vect = vectorizer.transform([cleaned_review])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect).max()

        history.append({
            "review": review,
            "prediction": pred,
            "confidence": round(prob * 100, 2)
        })

    return render_template('index.html', history=history)

if __name__ == "__main__":
    app.run(debug=True)
