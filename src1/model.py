from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

# Charger le dataset
df = pd.read_csv('src1/data/dataset_sentiment.csv')

#Prétraiter le texte avec le TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Appliquer le vectorizer sur les commentaires
X = vectorizer.fit_transform(df['comment'])
y_transition = df['transition_label']
y_sentiment = df['global_sentiment']
#test
#iviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train_transition, y_test_transition, y_train_sentiment, y_test_sentiment = train_test_split(
    X, y_transition, y_sentiment, test_size=0.2, random_state=42)

#Entraîner deux modèles RandomForest : un pour la transition et un pour le sentiment global
model_transition = RandomForestClassifier(n_estimators=1000, random_state=42)
model_sentiment = RandomForestClassifier(n_estimators=1000, random_state=42)

# Entraîner le modèle pour la transition de sentiment
model_transition.fit(X_train, y_train_transition)

# Entraîner le modèle pour le sentiment global
model_sentiment.fit(X_train, y_train_sentiment)

#Évaluer les performances des modèles
y_pred_transition = model_transition.predict(X_test)
y_pred_sentiment = model_sentiment.predict(X_test)

print(f"Transition Accuracy: {accuracy_score(y_test_transition, y_pred_transition)}")
print(f"Transition Classification Report: \n{classification_report(y_test_transition, y_pred_transition)}")

print(f"Sentiment Accuracy: {accuracy_score(y_test_sentiment, y_pred_sentiment)}")
print(f"Sentiment Classification Report: \n{classification_report(y_test_sentiment, y_pred_sentiment)}")

#Sauvegarder les modèles et le vectorizer
joblib.dump(model_transition, 'src1/models/sentiment_transition_model_rf.joblib')
joblib.dump(model_sentiment, 'src1/models/sentiment_model_rf.joblib')
joblib.dump(vectorizer, 'src1/models/vectorizer_tfidf.joblib')

