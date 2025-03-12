from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Exemple de données
comments = [
    ("Tiako be ity vokatra ity!", "positif"),
    ("Tena ratsy be ilay service, tsy manome fahafaham-po.", "négatif"),
    ("Afaka manatsara ny kalitao, fa mety ihany.", "neutre"),
    ("Mahafinaritra ny traikefa niainako tamin'ny fampiasana azy!", "positif"),
    ("Tsy mety ilay fanaterana, tara be!", "négatif"),
    ("Tsy dia ratsy, fa tsy dia tsara loatra koa.", "neutre"),
    ("Tena tsara, manoro hevitra ny hafa hampiasa azy!", "positif"),
    ("Diso fanantenana tanteraka, tsy hanome naoty tsara aho.", "négatif"),
    ("Misy zavatra tokony ahitsy, fa afaka ekena amin'ny ankapobeny.", "neutre")
]

df = pd.DataFrame(comments, columns=["commentaire", "label"])

# Nettoyage des données
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r'[^a-zA-Zà-ùÀ-Ù]+', ' ', texte)
    stopwords_malagasy = ["ny", "fa", "tsy", "dia", "izany", "ve", "no", "hoe"]  # Liste de stopwords enrichie
    mots = texte.split()
    mots_filtres = [mot for mot in mots if mot not in stopwords_malagasy]
    return " ".join(mots_filtres)

df["commentaire_nettoye"] = df["commentaire"].apply(nettoyer_texte)

# Vectorisation avec TF-IDF et n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Unigrams, Bigrams, Trigrams
X_tfidf = vectorizer.fit_transform(df["commentaire_nettoye"])

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["label"], test_size=0.2, random_state=42)

# Random Forest Optimisé avec Pruning et Entropie
class RandomForestCustom:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def calculer_entropy(self, y):
        total = len(y)
        counts = Counter(y)
        entropy = 0
        for count in counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        return entropy

    def fit(self, X, y):
        for _ in range(self.n_trees):
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample = X[indices]
            y_sample = y.iloc[indices].values
            tree = self.build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)

    def build_tree(self, X, y, depth):
        # Convertir X en matrice dense pour éviter l'erreur
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        # Calculer l'entropie pour chaque feature (colonne)
        entropies = [self.calculer_entropy(y[X_dense[:, i] > 0]) for i in range(X_dense.shape[1])]
        best_feature = np.argmin(entropies)  # Choisir la feature avec la plus faible entropie

        left_indices = X_dense[:, best_feature] > 0
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return Counter(y).most_common(1)[0][0]

        left_tree = self.build_tree(X_dense[left_indices], y[left_indices], depth + 1)
        right_tree = self.build_tree(X_dense[right_indices], y[right_indices], depth + 1)

        return {"feature": best_feature, "left": left_tree, "right": right_tree}

    def predict_tree(self, tree, x):
        # Convertir x en vecteur dense
        x_dense = x.toarray().flatten() if hasattr(x, "toarray") else x.flatten()

        if "feature" in tree:
            if x_dense[tree["feature"]] > 0:
                return self.predict_tree(tree["left"], x)
            else:
                return self.predict_tree(tree["right"], x)
        return tree

    def predict(self, X):
        predictions = np.array([self.predict_tree(tree, x) for tree in self.trees for x in X])
        predictions = predictions.reshape(len(self.trees), -1).T
        return [Counter(pred).most_common(1)[0][0] for pred in predictions]
    
# Entraînement du modèle
rf = RandomForestCustom(n_trees=300, max_depth=50, min_samples_split=2)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# Sauvegarde du modèle
joblib.dump((rf, vectorizer), "src2/models/random_forest_malagasy.pkl")

# Test du modèle
def tester_commentaire(commentaire):
    rf_model, vectorizer = joblib.load("src2/models/random_forest_malagasy.pkl")
    commentaire_nettoye = nettoyer_texte(commentaire)
    vecteur = vectorizer.transform([commentaire_nettoye])
    prediction = rf_model.predict(vecteur)
    return prediction[0]

commentaire_test = "tsy tiako be ilay vokatra!"
resultat = tester_commentaire(commentaire_test)
print(f"Le commentaire '{commentaire_test}' est classé comme : {resultat}")
