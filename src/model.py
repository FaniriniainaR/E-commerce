import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from utils import save_model, load_dataset

# Charger les données
data = load_dataset()

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(
    data["Commentaire"], data["Label"], test_size=0.2, random_state=42
)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3), stop_words=None)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialiser et entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=300, max_depth=50, min_samples_split=5, random_state=42)
model.fit(X_train_vec, y_train)

# Sauvegarde du modèle
save_model(model, vectorizer)
print("Modèle entraîné et sauvegardé avec succès !")

