import asyncio
import random
import pandas as pd
from deep_translator import GoogleTranslator

# 📌 Charger le dataset
file_path = "test.csv"  # Remplace par le chemin vers ton fichier CSV
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

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

# Listes pour stocker résultats et erreurs
textes_malagasy = []
erreurs = []
output_file = "DataSetMalgache.csv"  # Chemin de sortie

# Fonction asynchrone pour traiter une ligne
async def process_row(index, row):
    texte = row["text_clean"]
    sentiment = row["sentiment"].lower()  # En minuscules pour assurer la correspondance
    sentiment_fr = sentiment_mapping.get(sentiment, sentiment)
    try:
        # Exécuter la traduction dans un thread séparé pour ne pas bloquer la boucle asyncio
        traduction = await asyncio.to_thread(translator.translate, texte)
        print(f"Traitement {index+1}")
        if not traduction:
            raise ValueError("Traduction retournée vide")
    except Exception as e:
        print(f"❌ Erreur sur '{texte[:50]}...' : {e}")
        traduction = "Erreur de traduction"
        erreurs.append(texte)
    # Utiliser asyncio.sleep() pour attendre sans bloquer la boucle événementielle
    await asyncio.sleep(random.uniform(1, 1.5))
    return traduction, sentiment_fr

# Fonction asynchrone principale qui traite toutes les lignes
async def main():
    tasks = []
    for index, row in df.iterrows():
        tasks.append(process_row(index, row))
    # Exécuter les tâches en parallèle
    results = await asyncio.gather(*tasks)
    return results

if _name_ == "__main__":
    # Lancer la boucle événementielle asyncio
    results = asyncio.run(main())

    # Mettre à jour le DataFrame avec les résultats
    df["text_malagasy"], df["sentiment"] = zip(*results)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Traduction terminée ! Dataset enregistré sous '{output_file}'")

    # Sauvegarder les erreurs dans un fichier
    if erreurs:
        with open("erreurs_traduction.txt", "w", encoding="utf-8") as f:
            for err in erreurs:
                f.write(err + "\n")
        print(f"⚠️ {len(erreurs)} commentaires n'ont pas pu être traduits. Vérifie 'erreurs_traduction.txt'.")