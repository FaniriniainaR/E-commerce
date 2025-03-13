import re
from typing import List, Dict
from collections import Counter

import numpy as np

STOPWORDS_MALAGASY = {
    "ny", "fa", "dia", "izany", "ve", "no", "hoe", "de", "na", "a", "i", "an", "amin", "izao", "izay", "izany", "izy", "izaho",
    "satria", "efa", "mbola", "tokony", "ho", "an", "ary", "ka", "misy", "ireo", "ireny", "ireo",
    "nandritra", "ireny",
}

def nettoyer_texte(texte: str) -> str:
    """
    Nettoie le texte malgache en appliquant :
    - Conversion en minuscules
    - Suppression des caractères spéciaux
    - Filtrage des stopwords
    
    Args:
        texte (str): Texte à nettoyer
        
    Returns:
        str: Texte nettoyé
    """
    texte = texte.lower()
    texte = re.sub(r'[^a-zA-Zà-ùÀ-Ù\s]', ' ', texte)
    mots = texte.split()
    mots_filtres = [mot for mot in mots if mot not in STOPWORDS_MALAGASY]
    return " ".join(mots_filtres)

def calculer_entropy(y: np.ndarray) -> float:
    """
    Calcule l'entropie d'un ensemble de labels.
    
    Args:
        y (np.ndarray): Tableau des labels
        
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