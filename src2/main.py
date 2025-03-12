import joblib
from text_process import nettoyer_texte
from sklearn.model_selection import train_test_split
from data_utils import creer_dataset, vectoriser_donnees
from randomForestC import RandomForestCustom

def main():
    """
    Fonction principale qui orchestre l'ensemble du processus d'apprentissage.
    """
    # Création et vectorisation du dataset
    comments = [
        ("Tiako be ity vokatra ity!", "positif"),
        ("Tena ratsy be ilay service, tsy manome fahafaham-po.", "négatif"),
        ("Afaka manatsara ny kalitao, fa mety ihany.", "neutre"),
    ]
    
    df = creer_dataset(comments)
    X_tfidf, vectorizer = vectoriser_donnees(df)
    
    # Séparation données train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, df["label"], test_size=0.2, random_state=42
    )
    
    # Entraînement du modèle
    rf = RandomForestCustom(
        n_trees=300,          # Nombre d'arbres dans la forêt
        max_depth=50,         # Profondeur maximale des arbres
        min_samples_split=2,  # Nombre minimum d'échantillons pour une division
        pruning_threshold=0.1 # Seuil d'entropie pour le pruning
    )
    rf.fit(X_train, y_train)
    
    # Sauvegarde du modèle
    joblib.dump((rf, vectorizer), "src2/models/random_forest_malagasy.pkl")

def tester_commentaire(commentaire: str) -> str:
    """
    Test un nouveau commentaire avec le modèle sauvegardé.
    
    Args:
        commentaire (str): Commentaire à analyser
        
    Returns:
        str: Label prédit (positif, négatif ou neutre)
    """
    rf_model, vectorizer = joblib.load("src2/models/random_forest_malagasy.pkl")
    commentaire_nettoye = nettoyer_texte(commentaire)
    vecteur = vectorizer.transform([commentaire_nettoye])
    prediction = rf_model.predict(vecteur)
    return prediction[0]

if __name__ == "__main__":
    main()
    
    # Exemple de test
    commentaire_test = "tsy tiako be ilay vokatra!"
    resultat = tester_commentaire(commentaire_test)
    print(f"Le commentaire '{commentaire_test}' est classé comme : {resultat}")