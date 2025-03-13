import time
import random
import pandas as pd
from deep_translator import GoogleTranslator

# 📌 Charger le deuxième dataset (celui avec "Text" et "Sentiment")
try:
    df2 = pd.read_csv("src2/data/test2.csv", encoding="utf-8")
except UnicodeDecodeError:
    df2 = pd.read_csv("src2/data/test2.csv", encoding="ISO-8859-1") # Si UTF-8 échoue, essaie ISO-8859-1

# 📌 Initialiser le traducteur pour traduire en malgache
translator = GoogleTranslator(source='auto', target='mg')

# 📌 Traduire les textes du deuxième dataset
texts_malagasy_2 = []
erreurs_2 = []

for i, row in df2.iterrows():
    texte = row["Text"]
    sentiment = row["Sentiment"]  # Garder le sentiment intact
    
    try:
        traduction = translator.translate(texte)
        if not traduction:
            raise ValueError("Traduction retournée vide")
    except Exception as e:
        print(f"❌ Erreur sur '{texte[:50]}...' : {e}")
        traduction = "Erreur de traduction"
        erreurs_2.append(texte)

    texts_malagasy_2.append((traduction, sentiment))

    # 📌 Pause aléatoire pour éviter le blocage
    time.sleep(random.uniform(3, 6))

# 📌 Ajouter les traductions et le sentiment au deuxième dataset
df2["Text_malagasy"], df2["Sentiment"] = zip(*texts_malagasy_2)

# 📌 Ajouter les colonnes manquantes pour correspondre au premier dataset
df2["Source"] = df2["Source"]
df2["Date/Time"] = pd.NA
df2["User ID"] = pd.NA
df2["Location"] = pd.NA
df2["Confidence Score"] = pd.NA

# 📌 Charger le premier dataset déjà traduit
df1 = pd.read_csv("src2/data/DataSetMalgache.csv", encoding="utf-8")

# 📌 Ajuster les colonnes pour que ce soit similaire à df2
df1 = df1.rename(columns={'text_malagasy': 'Text', 'sentiment': 'Sentiment'})  # Adapter les colonnes du premier dataset
df1['Source'] = pd.NA
df1['Date/Time'] = pd.NA
df1['User ID'] = pd.NA
df1['Location'] = pd.NA
df1['Confidence Score'] = pd.NA

# 📌 Combiner les deux datasets
df_combined = pd.concat([df1, df2], ignore_index=True, sort=False)

# 📌 Sauvegarder le dataset combiné
output_combined_file = "src2/data/dataset_combine_traduits.csv"
df_combined.to_csv(output_combined_file, index=False, encoding="utf-8")
print(f"✅ Datasets combinés et enregistrés sous '{output_combined_file}'")

# 📌 Sauvegarder les erreurs pour les retraduire plus tard
if erreurs_2:
    with open("src2/data/erreurs_traduction_2.txt", "w", encoding="utf-8") as f:
        for err in erreurs_2:
            f.write(err + "\n")
    print(f"⚠️ {len(erreurs_2)} textes n'ont pas pu être traduits dans le deuxième dataset. Vérifie 'erreurs_traduction_2.txt'.")
