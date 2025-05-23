import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Configuration ---
# Set page configuration for better appearance
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the model and vectorizer to avoid reloading on every rerun
def load_resources():
    """Loads the trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load('models/multinomial_naive_bayes_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'multinomial_naive_bayes_model.pkl' and 'tfidf_vectorizer.pkl' are in the 'models/' directory.")
        st.stop() # Stop the app execution if files are missing
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

model, vectorizer = load_resources()

# --- Text Preprocessing Function (consistent with train_model.py) ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing URLs,
    numbers, punctuation, applying stemming, and removing stopwords.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words] # Stemming and stop word removal
    return ' '.join(tokens)

# --- Classification Function ---
def classify_email(email_content):
    """
    Classifies a single email content as 'spam' or 'ham' using the loaded model.
    """
    preprocessed_email = preprocess_text(email_content)
    # Transform the preprocessed email using the loaded vectorizer
    email_vectorized = vectorizer.transform([preprocessed_email])

    prediction = model.predict(email_vectorized)
    return 'spam' if prediction[0] == 1 else 'ham'

# --- Streamlit UI ---
st.title("📧 Email Spam Classifier")
st.markdown("Enter the content of an email below to classify it as **Spam** or **Ham** (Not Spam).")

# Text area for user input
email_input = st.text_area("Enter email content here:", height=250, placeholder="Type or paste your email content...")

# Button to trigger classification
if st.button("Classify Email"):
    if email_input:
        with st.spinner("Classifying..."):
            classification_result = classify_email(email_input)

        if classification_result == 'spam':
            st.error(f"Prediction: This email is **{classification_result.upper()}**! ⚠️")
            st.balloons() # Add some fun for spam detection
        else:
            st.success(f"Prediction: This email is **{classification_result.upper()}**! ✅")
    else:
        st.warning("Please enter some email content to classify.")

st.markdown("---")
st.markdown("Developed using Python, scikit-learn, NLTK, and Streamlit.")
st.markdown("Ensure you have run `train_model.py` to generate the model files in the `models/` directory.")

# Optional: Add a sidebar for more info or settings
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Multinomial Naive Bayes model trained on a dataset of emails "
    "to classify new emails as either spam or ham. Text preprocessing includes "
    "lowercasing, removing URLs, numbers, punctuation, stop words, and stemming."
)