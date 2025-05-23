import streamlit as st
import joblib
import re
import string
import nltk # Ensure nltk is imported
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Configuration ---
st.set_page_config(
    page_title="Email Analysis System",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- NLTK Data Download Check & Download ---
# This block runs at the very start of the script.
# @st.cache_resource ensures it only executes once across reruns on Streamlit Cloud.
@st.cache_resource
def check_and_download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt') # Also check for 'punkt'
        st.success("NLTK data (stopwords, punkt) already available.")
    except nltk.downloader.DownloadError:
        st.warning("NLTK data not found. Downloading 'stopwords' and 'punkt'...")
        nltk.download('stopwords')
        nltk.download('punkt')
        st.success("NLTK data (stopwords, punkt) downloaded successfully!")
    except Exception as e:
        st.error(f"An error occurred during NLTK data check/download: {e}")
        st.stop() # Stop if essential data can't be obtained

# Execute the download check immediately
check_and_download_nltk_data()

# --- Text Preprocessing Function (consistent across all scripts) ---
# Now, stop_words can be safely defined as the NLTK data is guaranteed to be present.
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) # This line will now work

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

# --- Load Models and Vectorizers ---
# (Keep this function as is)
@st.cache_resource
def load_all_resources():
    # ... (rest of your load_all_resources function) ...
    try:
        # Classifier resources
        classifier_model = joblib.load('models/multinomial_naive_bayes_model.pkl')
        classifier_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

        # Clustering resources
        cluster_model = joblib.load('models/kmeans_clusterer.pkl')
        cluster_vectorizer = joblib.load('models/tfidf_vectorizer_for_cluster.pkl')

        return classifier_model, classifier_vectorizer, cluster_model, cluster_vectorizer
    except FileNotFoundError as e:
        st.error(f"Error: One or more model files not found. Please ensure all required .pkl files are in the 'models/' directory. "
                 f"Have you run 'train_classifier.py' and 'train_clusterer.py'? Detailed error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()
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