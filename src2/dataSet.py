import time
import random
import pandas as pd
from deep_translator import GoogleTranslator

# 📌 Charger le dataset
file_path = "src2/data/test.csv"  # Remplace par le chemin vers ton fichier CSV

# Essayer de charger le fichier avec différents encodages si nécessaire
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Si UTF-8 échoue, essaie ISO-8859-1

# 📌 Vérifier que les colonnes nécessaires existent
if "text" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("Le fichier CSV ne contient pas les colonnes 'text' et 'sentiment'. Vérifie le format.")

# 📌 Nettoyer et sélectionner les colonnes
df = df[["text", "sentiment"]].dropna().reset_index(drop=True)
df["text_clean"] = df["text"].str.strip()  # Nettoyer les espaces superflus dans les textes

# 📌 Mapping des sentiments en français
sentiment_mapping = {
    "positive": "positif",
    "neutral": "neutre",
    "negative": "négatif"
}

# 📌 Initialiser le traducteur
translator = GoogleTranslator(source='auto', target='mg')

# 📌 Traduire les commentaires en malgache
textes_malagasy = []
erreurs = []
output_file = "src2/data/DataSetMalgache.csv"  # Remplace par le chemin où tu veux enregistrer le fichier

for i, row in df.iterrows():
    texte = row["text_clean"]
    sentiment = row["sentiment"].lower()  # Mettre en minuscules pour assurer une correspondance

    # Changer le sentiment en français en utilisant le dictionnaire
    sentiment_fr = sentiment_mapping.get(sentiment, sentiment)  # Utiliser le sentiment d'origine si non trouvé
    
    try:
        traduction = translator.translate(texte)
        if not traduction:
            raise ValueError("Traduction retournée vide")
    except Exception as e:
        print(f"❌ Erreur sur '{texte[:50]}...' : {e}")
        traduction = "Erreur de traduction"
        erreurs.append(texte)

    textes_malagasy.append((traduction, sentiment_fr))  # Utiliser le sentiment en français

    # 📌 Pause aléatoire pour éviter le blocage
    time.sleep(random.uniform(1, 1.5))
    print(i)

# 📌 Sauvegarde finale dans le fichier CSV
df["text_malagasy"], df["sentiment"] = zip(*textes_malagasy)
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Traduction terminée ! Dataset enregistré sous '{output_file}'")

# 📌 Sauvegarder les erreurs pour les retraduire plus tard
if erreurs:
    with open("src2/data/erreurs_traduction.txt", "w", encoding="utf-8") as f:
        for err in erreurs:
            f.write(err + "\n")
    print(f"⚠️ {len(erreurs)} commentaires n'ont pas pu être traduits. Vérifie 'erreurs_traduction.txt'.")
