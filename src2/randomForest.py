import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import Optional, Tuple, List
import numpy as np
from sklearn.metrics import classification_report

# Fonctions d'utilis.py (copiées pour complétude)
def load_dataset(filepath="src/data/dataset.csv"):
    return pd.read_csv(filepath)

def save_model(model, vectorizer, filepath="src/models/sentiment_model_rf.joblib"):
    joblib.dump({"model": model, "vectorizer": vectorizer}, filepath)

def load_model(filepath="src/models/sentiment_model_rf.joblib"):
    if os.path.exists(filepath):
        data = joblib.load(filepath)
        return data["model"], data["vectorizer"]
    return None, None

class Node:
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomForestClassifierCustom:
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, n_jobs: int = -1, seed: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.seed = seed
        self.trees = []
        if seed is not None:
            np.random.seed(seed)
        print("✅ RandomForestClassifierCustom initialisé")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifierCustom':
        try:
            print("🔄 Début de l'entraînement...")
            self.trees = []
            for i in range(self.n_estimators):
                X_boot, y_boot = self._bootstrap_sample(X, y)
                tree = self._fit_tree(X_boot, y_boot, self.max_depth, self.min_samples_split)
                self.trees.append(tree)
                if i % 10 == 0:
                    print(f"🌳 Arbre {i+1}/{self.n_estimators} créé")
            print("✅ Entraînement terminé")
            return self
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement: {str(e)}")
            raise

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            n_samples = X.shape[0]
            idxs = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
            return X[idxs], y[idxs]
        except Exception as e:
            print(f"❌ Erreur dans _bootstrap_sample: {str(e)}")
            raise

    def _fit_tree(self, X: np.ndarray, y: np.ndarray, max_depth: Optional[int],
                  min_samples_split: int) -> Node:
        try:
            n_samples, n_features = X.shape
            if len(set(y)) == 1 or max_depth == 0:
                return Node(value=Counter(y).most_common(1)[0][0])
            
            feature_idxs = np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False)
            best_split = None
            best_gain = -1
            best_left_idxs, best_right_idxs = None, None
            
            for idx in feature_idxs:
                column_data = X[:, idx]
                thresholds = np.unique(column_data)
                for threshold in thresholds:
                    left_idxs = column_data <= threshold
                    right_idxs = ~left_idxs
                    if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                        continue
                    gain = self._information_gain(y, left_idxs, right_idxs)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {"feature": idx, "threshold": threshold}
                        best_left_idxs = left_idxs
                        best_right_idxs = right_idxs
            
            if best_gain == -1:
                return Node(value=Counter(y).most_common(1)[0][0])
            
            left = self._fit_tree(X[best_left_idxs], y[best_left_idxs],
                                max_depth-1 if max_depth else None, min_samples_split)
            right = self._fit_tree(X[best_right_idxs], y[best_right_idxs],
                                 max_depth-1 if max_depth else None, min_samples_split)
            
            return Node(feature=best_split["feature"], threshold=best_split["threshold"],
                       left=left, right=right)
        except Exception as e:
            print(f"❌ Erreur dans _fit_tree: {str(e)}")
            raise

    def _information_gain(self, y: np.ndarray, left_idxs: np.ndarray, right_idxs: np.ndarray) -> float:
        try:
            parent_entropy = self._entropy(y)
            n_parent = len(y)
            n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
            if n_left == 0 or n_right == 0:
                return 0
            e_left = self._entropy(y[left_idxs])
            e_right = self._entropy(y[right_idxs])
            return parent_entropy - (n_left / n_parent) * e_left - (n_right / n_parent) * e_right
        except Exception as e:
            print(f"❌ Erreur dans _information_gain: {str(e)}")
            raise

    def _entropy(self, y: np.ndarray) -> float:
        try:
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum(np.where(ps > 0, ps * np.log2(ps), 0))
        except Exception as e:
            print(f"❌ Erreur dans _entropy: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            print("🔮 Début des prédictions...")
            predictions = np.array([self._predict(inputs) for inputs in X])
            print("✅ Prédictions terminées")
            return predictions
        except Exception as e:
            print(f"❌ Erreur pendant la prédiction: {str(e)}")
            raise

    def _predict(self, inputs: np.ndarray) -> int:
        try:
            predictions = [self._traverse_tree(inputs, tree) for tree in self.trees]
            return Counter(predictions).most_common(1)[0][0]
        except Exception as e:
            print(f"❌ Erreur dans _predict: {str(e)}")
            raise

    def _traverse_tree(self, x: np.ndarray, tree: Node) -> int:
        try:
            if tree.value is not None:
                return tree.value
            if x[tree.feature] <= tree.threshold:
                return self._traverse_tree(x, tree.left)
            return self._traverse_tree(x, tree.right)
        except Exception as e:
            print(f"❌ Erreur dans _traverse_tree: {str(e)}")
            raise

class SentimentAnalyzer:
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None
        )
        self.model = RandomForestClassifierCustom(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        self.is_trained = False
        self.sentiment_mapping = {
            0: "négatif",
            1: "neutre",
            2: "positif"
        }
        print("✅ SentimentAnalyzer initialisé")

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        try:
            print("📝 Prétraitement des textes...")
            return self.vectorizer.fit_transform(texts)
        except Exception as e:
            print(f"❌ Erreur dans preprocess_text: {str(e)}")
            raise

    def train(self, texts: List[str], sentiments: List[int], test_size: float = 0.2):
        try:
            print("🔄 Début de l'entraînement...")
            # Prétraitement des données
            X_preprocessed = self.preprocess_text(texts)
            
            # Séparation en jeux d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(
                X_preprocessed, sentiments, test_size=test_size, random_state=42
            )
            
            # Entraînement du modèle
            self.model.fit(X_train, y_train)
            
            # Évaluation
            y_pred = self.model.predict(X_test)
            print("📊 Rapport de classification sur le jeu de test:")
            print(classification_report(y_test, y_pred))
            
            self.is_trained = True
            print("✅ Entraînement terminé")
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement: {str(e)}")
            raise

    def predict(self, texts: List[str]) -> List[str]:
        try:
            if not self.is_trained:
                raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
            print("🔮 Début des prédictions...")
            # Prétraitement des textes
            X_preprocessed = self.preprocess_text(texts)
            
            # Prédiction
            predictions = self.model.predict(X_preprocessed)
            print("✅ Prédictions terminées")
            
            return [self.sentiment_mapping[pred] for pred in predictions]
        except Exception as e:
            print(f"❌ Erreur pendant la prédiction: {str(e)}")
            raise

    def save(self, model_path: str = "src/models/sentiment_model_rf.joblib"):
        try:
            print("💾 Sauvegarde du modèle...")
            save_model(self.model, self.vectorizer, model_path)
            print("✅ Modèle sauvegardé")
        except Exception as e:
            print(f"❌ Erreur pendant la sauvegarde: {str(e)}")
            raise

    @staticmethod
    def load(model_path: str = "src/models/sentiment_model_rf.joblib"):
        try:
            print("📥 Chargement du modèle...")
            model, vectorizer = load_model(model_path)
            analyzer = SentimentAnalyzer()
            analyzer.model = model
            analyzer.vectorizer = vectorizer
            analyzer.is_trained = True
            print("✅ Modèle chargé")
            return analyzer
        except Exception as e:
            print(f"❌ Erreur pendant le chargement: {str(e)}")
            raise

# Exemple d'utilisation
if __name__ == "__main__":
    try:
        print("🚀 Début du programme")
        
        # Création d'un exemple de données
        commentaires = [
            "J'adore ce produit, il est incroyable !",
            "Ce produit est moyen, rien de spécial.",
            "Je déteste ce produit, c'est une arnaque.",
            "Très satisfait de l'achat, je le recommande",
            "Produit correct, mais pas exceptionnel.",
            "Dommage d'avoir acheté ce produit.",
            "Parfait ! Exactement ce que je cherchais.",
            "Rien à dire, produit standard.",
            "Déçu de la qualité du produit."
        ]
        
        sentiments = [2, 1, 0, 2, 1, 0, 2, 1, 0]  # 2=positif, 1=neutre, 0=négatif
        
        # Création et entraînement du modèle
        analyzer = SentimentAnalyzer()
        analyzer.train(commentaires, sentiments)
        
        # Test des prédictions
        nouveaux_commentaires = [
            "Produit excellent, je suis très satisfait !",
            "Rien à dire, produit moyen.",
            "Je suis déçu de cette expérience."
        ]
        sentiments_predits = analyzer.predict(nouveaux_commentaires)
        print("\nRésultats des prédictions:")
        for texte, sentiment in zip(nouveaux_commentaires, sentiments_predits):
            print(f"Texte: {texte}")
            print(f"Sentiment prédit: {sentiment}\n")
        
        # Sauvegarde du modèle
        analyzer.save()
        
        print("✅ Programme terminé avec succès")
    except Exception as e:
        print(f"❌ Erreur fatale: {str(e)}")