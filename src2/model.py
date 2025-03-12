import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from src2.randomForest import RandomForest
from utils import save_model, load_dataset
import gc

# Vectorisation des données
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    dtype=np.float32,
    stop_words=None
)

# Chargement des données
data = load_dataset()

# Séparation des données en train/test
X_train, X_test, y_train, y_test = train_test_split(
    data["Commentaire"], 
    data["Label"], 
    test_size=0.2, 
    random_state=42
)

# Vectorisation des données
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Conversion en format CSR pour l'efficacité mémoire
X_train_sparse = csr_matrix(X_train_vec)
X_test_sparse = csr_matrix(X_test_vec)

# Nettoyage temporaire
del X_train_vec, X_test_vec
gc.collect()

# Entraînement du modèle
model = RandomForest(
    n_estimators=100,
    max_depth=50,
    min_samples_split=5,
    seed=42
)

# Entraînement et sauvegarde
model.fit(X_train_sparse, y_train)
save_model(model, vectorizer)
print("Modèle entraîné et sauvegardé avec succès !")

