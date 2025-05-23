# 📧 Email Spam Classifier

This project implements an Email Spam Classifier using a Multinomial Naive Bayes algorithm. It features a text preprocessing pipeline, TF-IDF for feature extraction, and a user-friendly web interface built with Streamlit.

The goal is to accurately classify incoming emails as either "Ham" (legitimate) or "Spam" (unsolicited/malicious).

---

## ✨ Features

- **Text Preprocessing:** Cleans and normalizes email content (lowercasing, URL/number/punctuation removal, stop word removal, stemming).
- **TF-IDF Feature Extraction:** Converts text data into numerical features that the machine learning model can understand.
- **Multinomial Naive Bayes Model:** A robust and effective algorithm for text classification.
- **Imbalance Handling:** Utilizes `imbalanced-learn` (RandomOverSampler) to address imbalanced datasets, improving classification performance for both Ham and Spam.
- **Streamlit Web App:** Provides an interactive and intuitive user interface to test email classification in real-time.
- **Model Persistence:** Saves the trained model and vectorizer to avoid retraining every time the application is run.

---

## Project Structure

email_classifier/
├── venv/
├── data/
│ └── spam_ham_dataset.csv
├── models/
│ ├── multinomial_naive_bayes_model.pkl
│ └── tfidf_vectorizer.pkl
├── app.py
├── train_model.py
├── requirements.txt
└── .gitignore

---

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### 1. Clone the Repository (Optional, if starting from scratch)

If you're tracking your project with Git, you'd typically clone it. If you've been building it locally, ensure your directory structure matches the one above.

```bash
git clone <your-repo-url>
cd email_classifier
```
