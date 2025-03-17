# tester_model.py
import joblib
from text_process import nettoyer_texte

def tester_commentaire(commentaire: str) -> str:
    """
    Teste un commentaire avec le modèle entraîné.
    """
    rf_model, vectorizer = joblib.load("src2/models/random_forest_malagasy.pkl")
    commentaire_nettoye = nettoyer_texte(commentaire)
    vecteur = vectorizer.transform([commentaire_nettoye])
    prediction = rf_model.predict(vecteur)
    
    # Renvoyer le label sous forme de texte (positif, neutre, négatif)
    label_map = {-1: "Neutre", 0: "Négatif", 1: "Positif"}
    return label_map.get(prediction[0], "Inconnu")

# Test d'un commentaire
commentaire_test = "Niteny ilay famaritana hoe hotian'ny vadiko ilay izy kinajo tsia"
resultat = tester_commentaire(commentaire_test)
print(f"Le commentaire '{commentaire_test}' est classé comme : {resultat}")
