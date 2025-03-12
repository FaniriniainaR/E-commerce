import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from text_process import nettoyer_texte

def creer_dataset(comments: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Crée un DataFrame à partir d'une liste de commentaires étiquetés.
    
    Args:
        comments (List[Tuple[str, str]]): Liste de tuples (commentaire, label)
        
    Returns:
        pd.DataFrame: DataFrame contenant les données nettoyées
    """
    df = pd.DataFrame(comments, columns=["commentaire", "label"])
    df["commentaire_nettoye"] = df["commentaire"].apply(nettoyer_texte)
    return df

def vectoriser_donnees(df: pd.DataFrame, ngram_range=(1, 3)) -> Tuple:
    """
    Vectorise les données textuelles avec TF-IDF.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les commentaires nettoyés
        ngram_range (Tuple[int, int]): Plage des n-grams à utiliser
        
    Returns:
        Tuple: Matrice TF-IDF et vecteur de labels
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(df["commentaire_nettoye"])
    return X_tfidf, vectorizer