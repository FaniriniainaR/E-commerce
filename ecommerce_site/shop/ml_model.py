# test_model.py
import sys
import os
import joblib

# Ajouter le chemin vers le dossier "src2" à sys.path
sys.path.append('D:/ESMIA/IA/Exercice/E-commerce/src2')

# Importer le modèle après avoir ajouté le chemin
from randomForestC import *  # Utilise le bon nom de module pour ton modèle
from text_process import nettoyer_texte

# Charger le modèle et le vectorizer
chemin_modele = "D:/ESMIA/IA/Exercice/E-commerce/src2/models/random_forest_malagasy.pkl"

try:
    rf_model, vectorizer = joblib.load(chemin_modele)
    print("✅ Modèle IA chargé avec succès !")
except Exception as e:
    rf_model, vectorizer = None, None
    print(f"❌ Erreur lors du chargement du modèle : {e}")

def tester_commentaire(commentaire: str) -> str:
    """
    Teste un commentaire avec le modèle IA chargé.
    """
    if rf_model is None or vectorizer is None:
        return "Erreur"

    commentaire_nettoye = nettoyer_texte(commentaire)
    vecteur = vectorizer.transform([commentaire_nettoye])
    prediction = rf_model.predict(vecteur)
    
    # Mapping des labels
    label_map = {-1: "Neutre", 0: "Négatif", 1: "Positif"}
    return label_map.get(prediction[0], "Inconnu")

# Exemple de test avec un commentaire
commentaire_test = "Tena tsara ilay vokatra!"
resultat = tester_commentaire(commentaire_test)
print(f"Le commentaire '{commentaire_test}' est classé comme : {resultat}")
