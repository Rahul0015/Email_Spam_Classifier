import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
try:
    model = joblib.load('multinomial_naive_bayes_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run 'train_model.py' first.")
    exit()

# --- Text Preprocessing Function (same as in train_model.py) ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words] # Stemming and stop word removal
    return ' '.join(tokens)

def classify_email(email_content):
    """
    Classifies a single email content as 'spam' or 'ham'.
    """
    if not model or not vectorizer:
        print("Model or vectorizer not loaded. Cannot classify.")
        return None

    preprocessed_email = preprocess_text(email_content)
    # Transform the preprocessed email using the loaded vectorizer
    email_vectorized = vectorizer.transform([preprocessed_email])

    prediction = model.predict(email_vectorized)
    return 'spam' if prediction[0] == 1 else 'ham'

if __name__ == "__main__":
    print("\n--- Email Classifier ---")
    while True:
        email = input("Enter email content (or 'q' to quit): \n")
        if email.lower() == 'q':
            break

        classification = classify_email(email)
        if classification:
            print(f"This email is classified as: {classification.upper()}")
        print("-" * 30)