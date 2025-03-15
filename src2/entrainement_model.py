# entrainement_model.py
from typing import Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from randomForestC import RandomForestCustom
from text_process import nettoyer_texte
from sklearn.metrics import accuracy_score

def charger_dataset(csv_path: str) -> pd.DataFrame:
    """
    Charge un dataset à partir d'un fichier CSV et applique le prétraitement.
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Assurez-vous que les colonnes attendues existent
    if "Commentaire" not in df.columns or "Label" not in df.columns:
        raise KeyError("Le fichier CSV ne contient pas les colonnes attendues: 'Commentaire' et 'Label'")

    # Prétraitement des commentaires
    df["commentaire_nettoye"] = df["Commentaire"].apply(nettoyer_texte)

    # Vérifier les valeurs de la colonne "Label"
    if not all(df["Label"].isin([-1, 0, 1])):
        raise ValueError("Les labels de la colonne 'Label' doivent être -1, 0 ou 1.")
    
    return df

def vectoriser_donnees(df: pd.DataFrame, ngram_range=(1, 5)) -> Tuple:
    """
    Vectorise les commentaires avec TF-IDF.
    """
    if "commentaire_nettoye" not in df.columns:
        raise KeyError("La colonne 'commentaire_nettoye' est absente du DataFrame !")
    
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(df["commentaire_nettoye"])
    return X_tfidf, vectorizer

def main():
    """
    Fonction principale pour entraîner le modèle Random Forest.
    """
    csv_path = "src2/data/dataset.csv"  # Remplacez avec le chemin vers votre fichier CSV
    df = charger_dataset(csv_path)
    
    # Vectorisation des données
    X_tfidf, vectorizer = vectoriser_donnees(df)
    
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, df["Label"], test_size=0.2, random_state=42
    )
    
    # Initialisation et entraînement du modèle Random Forest personnalisé
    rf = RandomForestCustom(
        n_trees=1500,
        max_depth=30,
        min_samples_split=5,
        pruning_threshold=0.1
    )
    rf.fit(X_train, y_train)
    
    # Sauvegarde du modèle et du vectorizer
    joblib.dump((rf, vectorizer), "src2/models/random_forest_malagasy.pkl")
    print("Modèle entraîné et sauvegardé avec succès !")

    # Évaluation de la performance
    y_pred = rf.predict(X_test)
    print(f"Précision sur l'ensemble de test : {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
