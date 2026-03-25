# Automated Essay Scoring App

A Streamlit app that predicts essay scores using a trained machine learning pipeline.

## Features

- Essay score prediction
- Built-in sample essays for quick testing
- Basic writing stats (words, sentences, average sentence length)
- Quick quality feedback label

## Dataset

This project is based on the ASAP 2.0 dataset:

- https://www.kaggle.com/datasets/lburleigh/asap-2-0

## Project Files

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `essay_model.pkl` - Trained scoring model
- `tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer
- `scaler.pkl` - Saved feature scaler

## Local Run

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run app.py

## Notes

- Keep the model and preprocessing files in the project root.
- If predictions feel compressed in range, retraining with improved features and prompt-aware handling is recommended.
