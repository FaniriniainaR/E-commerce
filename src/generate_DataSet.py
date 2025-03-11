
import random
import pandas as pd

# Commentaires d'exemple
positif_comments = [
    "Tsara be ilay vokatra", "Faly amin'ny vokatra", "Tena tsara ilay vokatra",
    "Vokatra manara-penitra", "Tena mahafinaritra", "Izany no vokatra tiako",
    "Ilay vokatra dia tena ilaina", "Mahafaly ahy ilay izy", "faly aho tamin'ny vokatra", "Tsara fatohizo", "Tsy ratsy", "tsy de ratsy"
]

negatif_comments = [
    "Tsy mahafinaritra ilay vokatra", "Mampalahelo fa tsy niasa ilay izy",
    "Tena tsy nety mihitsy", "Vokatra ratsy", "Misy olana goavana amin'ity vokatra ity",
    "Vokatra tsy manara-penitra", "Ity vokatra ity dia tsy tonga amin'ny fotoana",
    "Tsy nandeha tsara ilay izy", "Masosotra"
]

neutre_comments = [
    "Vokatra mahazatra fotsiny", "Tsy nisy zavatra manokana tamin'ny vokatra",
    "Ilay vokatra dia toy ny hafa", "Ity vokatra ity dia mbola tsy voamarina",
    "Tsy misy zavatra mahagaga amin'ny vokatra", "Vokatra mety ho tsara, fa tsy manan-danja",
    "Izany vokatra izany dia tsy hoe ratsy, fa tsy manampy", "Vokatra an'ny marika hafa"
]

# Fonction pour générer des commentaires
def generate_comments(num_comments):
    data = []
    for _ in range(num_comments):
        sentiment = random.choice([1, 0, -1])  # Choisir un sentiment au hasard
        if sentiment == 1:
            comment = random.choice(positif_comments)
        elif sentiment == 0:
            comment = random.choice(negatif_comments)
        else:
            comment = random.choice(neutre_comments)
        data.append([comment, sentiment])
    
    return data

# Générer 1 million de commentaires
num_comments = 1000000
comments_data = generate_comments(num_comments)

# Convertir en DataFrame et sauvegarder en CSV
df = pd.DataFrame(comments_data, columns=["Commentaire", "Label"])
df.to_csv("src/data/dataset.csv", index=False, encoding='utf-8')

print("Fichier CSV créé avec succès : dataset.csv")
