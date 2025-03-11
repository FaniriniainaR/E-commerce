import random
import pandas as pd

# Liste de commentaires positifs, négatifs et neutres en malgache
positif_comments = [
    "Tsara be ilay vokatra, tena manampy ahy, azoko ampiasaina isan'andro.",
    "Faly amin'ny vokatra, mahafinaritra sy manampy amin'ny zavatra andavanandro.",
    "Tena tsara ilay vokatra, mahafinaritra ny fahombiazan'ny fiasa, tsy mbola nisy olana.",
    "Izany no vokatra tiako, tena mahasoa sy manampy amin'ny asa andavanandro.",
]

negatif_comments = [
    "Misy olana goavana amin'ity vokatra ity, tsy miasa araka ny nampoizina.",
    "Tsy nety mihitsy ilay izy, diso fanantenana be aho.",
    "Tsy mahafinaritra ilay vokatra, mitranga tsy tapaka ny fahadisoana.",
    "Vokatra ratsy, tsy nahafa-po mihitsy, tsy nandeha tsara ilay izy.",
]

neutre_comments = [
    "Vokatra mahazatra fotsiny, tsy misy zavatra mahagaga.",
    "Tsy nisy zavatra manokana tamin'ny vokatra, toy ny hafa.",
    "Ilay vokatra dia toy ny hafa, mety ho tsara fa tsy manan-danja.",
    "Ity vokatra ity dia mbola tsy voamarina, tsy misy zavatra manokana.",
]

# Fonction pour générer des commentaires avec des transitions de sentiment et sentiment global
def generate_data(num_comments=1000000):
    data = []
    labels = []
    global_sentiment = []

    for _ in range(num_comments):
        sentiment_order = random.choice(['positive_to_negative', 'negative_to_positive'])
        
        if sentiment_order == 'positive_to_negative':
            # Positive au début, négatif à la fin
            start = random.choice(positif_comments)
            end = random.choice(negatif_comments)
            comment = start + " " + end
            label = 1  # 1 pour positif au début et négatif à la fin
        else:
            # Négatif au début, positif à la fin
            start = random.choice(negatif_comments)
            end = random.choice(positif_comments)
            comment = start + " " + end
            label = 0  # 0 pour négatif au début et positif à la fin

        # Ajouter sentiment global basé sur la majorité des sentiments dans le commentaire
        if random.random() < 0.33:  # 33% chance to have a neutral comment
            sentiment = 'neutre'
        elif random.random() < 0.66:  # 33% chance to have a positive comment
            sentiment = 'positif'
        else:  # 34% chance to have a negative comment
            sentiment = 'negatif'

        data.append(comment)
        labels.append(label)
        global_sentiment.append(sentiment)

    return data, labels, global_sentiment

# Générer le dataset
comments, labels, sentiments = generate_data(1000000)

# Convertir en DataFrame pour l'enregistrement
df = pd.DataFrame({
    'comment': comments,
    'transition_label': labels,
    'global_sentiment': sentiments
})

# Sauvegarder le dataset en CSV
df.to_csv('src1/data/dataset_sentiment.csv', index=False)
