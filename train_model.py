import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving/loading models
import os # Import the os module for path manipulation

# Import for handling imbalanced data
from imblearn.over_sampling import RandomOverSampler

# --- 1. Load Data ---
# Make sure 'spam_ham_dataset.csv' is in the 'data/' directory relative to where this script is run
try:
    df = pd.read_csv('data/spam_ham_dataset.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print("Error: 'spam_ham_dataset.csv' not found. Please ensure it's in the 'data/' directory.")
    exit()

# Rename columns for consistency (adjust if your CSV has different names)
df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)

# Convert labels to numerical (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Check Initial Label Distribution ---
print("\n--- Initial Label Distribution ---")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True)) # Show as percentages


# --- 2. Text Preprocessing ---
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

print("\nPreprocessing text...")
df['preprocessed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete.")
print(df[['text', 'preprocessed_text']].head())


# --- 3. Feature Extraction (TF-IDF) ---
print("\nPerforming TF-IDF feature extraction...")
# Initialize TfidfVectorizer with an increased max_features
tfidf_vectorizer = TfidfVectorizer(max_features=10000) # Increased max_features
X = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
y = df['label']
print("TF-IDF feature extraction complete.")
print(f"Shape of TF-IDF matrix (X): {X.shape}")

# --- 4. Handle Imbalance (Oversampling) ---
print("\nBalancing dataset using RandomOverSampler...")
# Apply Random Oversampling to the minority class (Spam is '1')
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print(f"Original label distribution: {y.value_counts()}")
print(f"Resampled label distribution: {y_resampled.value_counts()}")
print(f"Resampled label distribution (normalized): {y_resampled.value_counts(normalize=True)}")

# --- 5. Train-Test Split (on resampled data) ---
print("\nSplitting data into training and testing sets...")
# Stratify by the resampled 'y' to maintain balance in train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
print(f"Training data shape after resampling: {X_train.shape}")
print(f"Testing data shape after resampling: {X_test.shape}")

# --- 6. Train Multinomial Naive Bayes Model ---
print("\nTraining Multinomial Naive Bayes model...")
mnb_model = MultinomialNB()
mnb_model.fit(X_train, y_train)
print("Model training complete.")

# --- 7. Evaluate Model ---
print("\nEvaluating model performance...")
y_pred = mnb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# --- Confirmation of Predictions on Test Set ---
# This helps confirm if the model is still predicting all one class
print("\n--- Sample Predictions on Test Set ---")
print("First 20 actual labels (y_test):", y_test.head(20).tolist())
print("First 20 predicted labels (y_pred):", y_pred[:20].tolist())
print("Count of predicted Ham (0) in test set:", list(y_pred).count(0))
print("Count of predicted Spam (1) in test set:", list(y_pred).count(1))

# --- 8. Save Model and Vectorizer ---
print("\nSaving the trained model and TF-IDF vectorizer...")

# Define the directory where models should be saved
models_dir = 'models'

# Create the 'models' directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Specify the full paths including the 'models/' directory
model_path = os.path.join(models_dir, 'multinomial_naive_bayes_model.pkl')
vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

joblib.dump(mnb_model, model_path)
joblib.dump(tfidf_vectorizer, vectorizer_path)
print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")
print("Model and vectorizer saved successfully into the 'models/' directory.")
