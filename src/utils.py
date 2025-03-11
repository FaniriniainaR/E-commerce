import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Charger le dataset
def load_dataset(filepath="src/data/dataset.csv"):
    return pd.read_csv(filepath)

# Sauvegarder un modèle
def save_model(model, vectorizer, filepath="src/models/sentiment_model_rf.joblib"):
    joblib.dump({"model": model, "vectorizer": vectorizer}, filepath)

# Charger un modèle
def load_model(filepath="src/models/sentiment_model_rf.joblib"):
    if os.path.exists(filepath):
        data = joblib.load(filepath)
        return data["model"], data["vectorizer"]
    return None, None
