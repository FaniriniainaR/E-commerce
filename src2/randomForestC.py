import numpy as np
from collections import Counter
from scipy import sparse
from typing import Union, Dict, Any

class RandomForestCustom:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=5, pruning_threshold=0.1):
        """
        Initialise le Random Forest avec paramètres d'optimisation.
        
        Args:
            n_trees (int): Nombre d'arbres dans la forêt
            max_depth (int): Profondeur maximale des arbres
            min_samples_split (int): Nombre minimum d'échantillons pour une division
            pruning_threshold (float): Seuil d'entropie pour le pruning
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.pruning_threshold = pruning_threshold
        self.trees = []
        
    def calculer_entropy(self, y):
        """
        Calcule l'entropie d'un ensemble de labels avec optimisation.
        
        Args:
            y (array-like): Tableau des labels
            
        Returns:
            float: Entropie calculée
        """
        total = len(y)
        counts = Counter(y)
        entropy = 0
        for count in counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        return entropy
        
    def fit(self, X, y):
        """
        Entraîne le modèle avec échantillonnage bootstrap et pruning.
        
        Args:
            X (array-like): Matrice des caractéristiques
            y (array-like): Vecteur des labels
        """
        for _ in range(self.n_trees):
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample = X[indices]
            y_sample = y.iloc[indices].values
            tree = self.build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)
            
    def build_tree(self, X, y, depth):
        """
        Construit un arbre de décision avec pruning basé sur l'entropie.
        
        Args:
            X (array-like): Matrice des caractéristiques
            y (array-like): Vecteur des labels
            depth (int): Profondeur courante de l'arbre
            
        Returns:
            dict ou str: Structure de l'arbre ou label
        """
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        
        if (depth >= self.max_depth or 
            len(set(y)) == 1 or 
            len(y) < self.min_samples_split or 
            self.calculer_entropy(y) < self.pruning_threshold):
            return Counter(y).most_common(1)[0][0]
            
        entropies = []
        for i in range(X_dense.shape[1]):
            mask = X_dense[:, i].astype(bool)
            if np.any(mask):
                subset_y = y[mask]
                entropies.append(self.calculer_entropy(subset_y))
            else:
                entropies.append(float('inf'))
                
        best_feature = np.argmin(entropies)
        
        left_indices = X_dense[:, best_feature].astype(bool)
        right_indices = ~left_indices
        
        if (np.sum(left_indices) < self.min_samples_split or 
            np.sum(right_indices) < self.min_samples_split):
            return Counter(y).most_common(1)[0][0]
            
        left_tree = self.build_tree(X_dense[left_indices], 
                                  y[left_indices], 
                                  depth + 1)
        right_tree = self.build_tree(X_dense[right_indices], 
                                   y[right_indices], 
                                   depth + 1)
        
        return {"feature": best_feature, 
                "left": left_tree, 
                "right": right_tree}
    
    def predict_tree(self, tree, x):
        """
        Prédit le label pour un échantillon donné à partir d'un arbre.
        
        Args:
            tree (dict ou int): Structure de l'arbre ou un label direct
            x (array-like): Échantillon à prédire
            
        Returns:
            int: Label prédit
        """
        x_dense = x.toarray().flatten() if hasattr(x, "toarray") else x.flatten()

        # Si on atteint une feuille (label directement retourné)
        if isinstance(tree, int):
            return tree
        
        # Vérifiez d'abord que l'arbre est un dictionnaire avant de chercher "feature"
        if isinstance(tree, dict) and "feature" in tree:
            if x_dense[tree["feature"]] > 0:
                return self.predict_tree(tree["left"], x)
            else:
                return self.predict_tree(tree["right"], x)
        
        # Si l'arbre ne correspond à aucun cas attendu, retourner un label par défaut
        return tree


    
    def predict(self, X):
        """
        Prédit les labels pour un ensemble d'échantillons.
        
        Args:
            X (array-like): Matrice des caractéristiques
            
        Returns:
            list: Liste des labels prédits
        """
        predictions = np.array([self.predict_tree(tree, x) for tree in self.trees for x in X])
        predictions = predictions.reshape(len(self.trees), -1).T
        return [Counter(pred).most_common(1)[0][0] for pred in predictions]