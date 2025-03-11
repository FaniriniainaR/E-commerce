from utils import load_model
import numpy as np

# Fonction pour gérer les longs commentaires
def predict_long_comment(comment, model, vectorizer):
    sentences = comment.split(".")  # Diviser en phrases
    sentences = [s.strip() for s in sentences if s.strip()]  # Nettoyer les phrases
    if not sentences:
        return 0  # Retourner neutre si le texte est vide
    predictions = [model.predict(vectorizer.transform([sentence]))[0] for sentence in sentences]
    return int(np.round(np.mean(predictions)))  # Moyenne des prédictions

# Charger le modèle
model, vectorizer = load_model()

if model and vectorizer:
    print("Modèle chargé avec succès !")

    # Tester avec un commentaire long
    comment = "Tsy ratsy kosa ilay izy anh"
    prediction = predict_long_comment(comment, model, vectorizer)
    
    # Afficher le résultat
    sentiments = {1: "POSITIF", 0: "NÉGATIF", -1: "NEUTRE"}
    print(f"Prédiction : {sentiments[prediction]}")

else:
    print("Modèle introuvable, entraînez-le avec `model.py`")

# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Charger les modèles et le vectorizer sauvegardés
# model_transition = joblib.load('src/models/sentiment_transition_model_rf.joblib')
# model_sentiment = joblib.load('src/models/sentiment_model1_rf.joblib')
# vectorizer = joblib.load('src/models/vectorizer_tfidf.joblib')

# # Fonction pour analyser le sentiment global et la transition de sentiment
# def analyze_comment(comment):
#     # Convertir le commentaire avec le vectorizer
#     X = vectorizer.transform([comment])
    
#     # Prédire le sentiment global
#     global_sentiment = model_sentiment.predict(X)[0]
    
#     # Prédire la transition de sentiment
#     sentiment_transition = model_transition.predict(X)[0]
    
#     # Afficher les résultats
#     sentiment_dict = {0: "Négatif", 1: "Positif", 2: "Neutre"}
#     transition_dict = {0: "Négatif au début, Positif à la fin", 1: "Positif au début, Négatif à la fin"}

#     print(f"Sentiment global : {sentiment_dict.get(global_sentiment, 'Inconnu')}")
#     print(f"Transition de sentiment : {transition_dict.get(sentiment_transition, 'Pas de retournement significatif')}")

# # Tester avec un commentaire
# commentaire_test = "Tsara be ilay vokatra, faly amin'ny vokatra. Fa rehefa nampiasaina, nisy olana goavana, tsy nety mihitsy."
# analyze_comment(commentaire_test)
