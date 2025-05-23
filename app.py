import streamlit as st
import joblib
import re
import string
import nltk # Ensure nltk is imported
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Configuration ---
st.set_page_config(
    page_title="Spam/Ham Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- NLTK Data Download Check & Download ---
# This block runs at the very start of the script when Streamlit loads the app.
# @st.cache_resource ensures it only executes once per deployment/session.
@st.cache_resource
def download_nltk_data_if_needed():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt') # Also check for 'punkt'
        st.success("NLTK data (stopwords, punkt) already available.")
    except nltk.downloader.DownloadError:
        st.warning("NLTK data not found. Attempting to download 'stopwords' and 'punkt'...")
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            st.success("NLTK data (stopwords, punkt) downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download NLTK data: {e}. Please check your internet connection or deployment environment.")
            st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during NLTK data check/download: {e}")
        st.stop()

# Execute the download check immediately
download_nltk_data_if_needed()

# --- Text Preprocessing Function ---
# Now, it's safe to define these global variables as NLTK data is guaranteed to be present.
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing URLs,
    numbers, punctuation, applying stemming, and removing stopwords.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the model and vectorizer to avoid reloading on every rerun
def load_classifier_resources():
    """Loads the trained classifier model and TF-IDF vectorizer."""
    try:
        # These paths are relative to where the app.py runs (project root on Streamlit Cloud)
        classifier_model = joblib.load('models/multinomial_naive_bayes_model.pkl')
        classifier_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return classifier_model, classifier_vectorizer
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found. Ensure 'multinomial_naive_bayes_model.pkl' and 'tfidf_vectorizer.pkl' are in the 'models/' directory. "
                 f"Have you run 'train_classifier.py' locally and committed the 'models/' folder to GitHub? Detailed error: {e}")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

# Load resources at app startup
classifier_model, classifier_vectorizer = load_classifier_resources()

# --- Classification Function ---
def classify_email(email_content):
    preprocessed_email = preprocess_text(email_content)
    # Transform the preprocessed email using the loaded vectorizer
    email_vectorized = classifier_vectorizer.transform([preprocessed_email])
    prediction = classifier_model.predict(email_vectorized)
    return 'spam' if prediction[0] == 1 else 'ham'

# --- Streamlit UI ---
st.title("📧 Basic Spam/Ham Email Classifier")
st.markdown("Enter the content of an email below to classify it as **Spam** or **Ham**.")

# Text area for user input
email_input = st.text_area("Enter email content:", height=250, placeholder="Type or paste your email content here...")

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
st.markdown("Developed using Python, scikit-learn, NLTK, imbalanced-learn, and Streamlit.")
st.markdown("Ensure you have run `python train_classifier.py` locally to generate the necessary model files in the `models/` directory, and committed them to your GitHub repository.")

# Optional: Add a sidebar for more info or settings
st.sidebar.header("About This System")
st.sidebar.info(
    "This is a basic email classifier that predicts whether an email is spam or ham. "
    "It uses a Multinomial Naive Bayes model trained with TF-IDF features."
)
