
import random
import pandas as pd

# Commentaires d'exemple
positif_comments = [
    "Tsara be ilay vokatra", "Faly amin'ny vokatra", "Tena tsara ilay vokatra",
    "Vokatra manara-penitra", "Tena mahafinaritra", "Izany no vokatra tiako",
    "Ilay vokatra dia tena ilaina", "Mahafaly ahy ilay izy", "faly aho tamin'ny vokatra", 
    "Tsara fatohizo", "Tsy ratsy", "tsy de ratsy", "Vokatra tsara", "Afam-po tanteraka", 
    "Afam-po", "Tena niavaka ilay vokatra", "Mankasitraka be ny vokatra", "Azo antoka fa hividy indray aho",
    "Izay ilaina rehetra ao aminy", "Mendrika ny vola laniana", "Fifaliana miverimberina", 
    "Tena faly amin'ny fanapahankevitra navoakako", "Vokatra tena mahafinaritra sy maharitra", 
    "Afaka manam-pahaizana amin'ity vokatra ity aho", "Ilay vokatra dia mifanaraka amin'ny andrasana",
    "Tena nahafinaritra ahy ny kalitaon'ny vokatra.",
    "Mora ampiasaina ary tena mahomby.",
    "Vokatra tena tsara ho an'ny vidiny.",
    "Tena nanampy ahy tamin'ny fiainako andavanandro.",
    "Vokatra manara-penitra sady maharitra.",
    "Tiako ny fomba nanamboarana azy, tena tsara ny antsipiriany.",
    "Tena mahafa-po ny serivisy sy ny vokatra.",
    "Tena faly aho fa nividy ity vokatra ity.",
    "Manoro hevitra ity vokatra ity amin'ny rehetra aho.",
    "Tsy hanenenana mihitsy raha mividy ity vokatra ity.",
    "Antenaina fa hanohy izao kalitao izao hatrany ianareo.",
    "Tena nanaitra ahy ny fahombiazan'ity vokatra ity.",
    "Ny fampiasana azy dia tena mora sy mahafinaritra.",
    "Vokatra tena tsara kalitao sady tena manome fahafaham-po.",
    "Tena manampy amin'ny fampandehanan'ny asa andavanandro.",
    "Vokatra tena manara-penitra sady maharitra ela.",
    "Tiako ny fomba nanamboarana azy, tena tsara ny antsipiriany.",
    "Tena mahafa-po ny serivisy sy ny vokatra.",
    "Tena faly aho fa nividy ity vokatra ity.",
    "Manoro hevitra ity vokatra ity amin'ny rehetra aho.",
    "Tsy hanenenana mihitsy raha mividy ity vokatra ity.",
    "Antenaina fa hanohy izao kalitao izao hatrany ianareo.",
    "Tena nampiaiky ahy ny haingam-pandehan'ity vokatra ity.",
    "Ny famolavolana azy dia tena kanto sy maoderina.",
    "Tena mora ny fikarakarana sy ny fanadiovana azy.",
    "Tena azo antoka ny fampiasana azy.",
]

negatif_comments = [
    "Tsy mahafinaritra ilay vokatra", "Mampalahelo fa tsy niasa ilay izy",
    "Tena tsy nety mihitsy", "Vokatra ratsy", "Misy olana goavana amin'ity vokatra ity",
    "Vokatra tsy manara-penitra", "Ity vokatra ity dia tsy tonga amin'ny fotoana",
    "Tsy nandeha tsara ilay izy", "Masosotra", "Tena ratsy", "Tsy tsara", "Masosotra", "Mankaleo", 
    "Mampalahelo", "Tsy tiako", "Tsy tiako ilay vokatra", "Vokatra ratsy", "Tsy mahafinaritra", 
    "Tsy nety", "Tsy niasa", "Tsy nety mihitsy", "Tsy nety mihitsy ilay vokatra", 
    "Tsy mahafam-po ilay vokatra", "Ratsy be ilay vokatra", "Tena tsy afam-po", "Tsy mahafam-po", 
    "Tena tsy mahafam-po", "Mila fanamboarana be ilay vokatra", "Tsy manan-danja ilay vokatra", 
    "Raha nisy hividy ahy dia haniry hanala", "Tsy mendrika ny vola laniana", 
    "Izany vokatra izany dia tena mampiady hevitra",
    "Tapaka ilay vokatra rehefa avy nampiasa azy indray mandeha monja.",
    "Tsy nifanaraka tamin'ny voalaza ny fampisehoana.",
    "Tena diso fanantenana aho tamin'ny faharetan'ilay vokatra.",
    "Tsy mandeha araka ny tokony ho izy ny fampiasa sasany.",
    "Tena nandany vola fotsiny aho tamin'ity vokatra ity.",
    "Nahasorena ahy ny fomba nanamboarana azy.",
    "Tsy azo antoka mihitsy ity vokatra ity.",
    "Mila manatsara ny kalitaon'ny fitaovana ianareo.",
    "Tokony ho tsara kokoa ny serivisy ho an'ny mpanjifa.",
    "Manantena aho fa hanao fanatsarana ianareo amin'ny ho avy.",
    "Tena diso fanantenana aho tamin'ny fampandehanan'ity vokatra ity."
    "Tsy nifanaraka tamin'ny voalaza ny fampisehoana.",
    "Tena diso fanantenana aho tamin'ny faharetan'ilay vokatra.",
    "Tsy mandeha araka ny tokony ho izy ny fampiasa sasany.",
    "Tena nandany vola fotsiny aho tamin'ity vokatra ity.",
    "Nahasorena ahy ny fomba nanamboarana azy.",
    "Tsy azo antoka mihitsy ity vokatra ity.",
    "Mila manatsara ny kalitaon'ny fitaovana ianareo.",
    "Tokony ho tsara kokoa ny serivisy ho an'ny mpanjifa.",
    "Manantena aho fa hanao fanatsarana ianareo amin'ny ho avy.",
    "Tena nahasorena ahy ny fahasarotan'ny fampiasana azy.",
    "Tsy nahafa-po ahy ny fomba nandefasana ilay vokatra.",
    "Tena nisy lesoka be ilay vokatra.",
    "Tsy mendrika ny vidiny mihitsy ilay vokatra.",
]

neutre_comments = [
    "Vokatra mahazatra fotsiny", "Tsy nisy zavatra manokana tamin'ny vokatra",
    "Ilay vokatra dia toy ny hafa", "Ity vokatra ity dia mbola tsy voamarina",
    "Tsy misy zavatra mahagaga amin'ny vokatra", "Vokatra mety ho tsara, fa tsy manan-danja",
    "Izany vokatra izany dia tsy hoe ratsy, fa tsy manampy", "Vokatra an'ny marika hafa",
    "Tsy mitombina ny hafainganam-pandehany", "Vokatra manara-penitra saingy mety mbola ilaina fanatsarana",
    "Mety tsara ho an'ny olona mitady vokatra mahazatra", "Vokatra sahaza ho an'ny fandehan'ny andavanandro",
    "Vokatra tsara, fa tsy misy fiavahana lehibe", "Vokatra afaka manampy fa tsy tena ilaina",
    "Mety ho tsara ho an'ny olona manokana", "Mahaliana fa tsy miavaka", "Vokatra azo ampiasaina, fa tsy tena manintona",
    "Tsy nahafaly ahy be, fa tsy ratsy ihany koa", "Raha tsy misy safidy hafa dia mety", 
    "Tsy tsara loatra, fa azo ekena",
    "Mety ho an'ny olona sasany ity vokatra ity.",
    "Tsy misy zavatra ratsy, fa tsy misy zavatra tsara be koa.",
    "Vokatra mahazatra, tsy misy miavaka.",
    "Mbola tsy nanandrana ny fampiasa rehetra aho.",
    "Mety mila fotoana vitsivitsy aho vao afaka manome hevitra feno.",
    "Miankina amin'ny filan'ny tsirairay ny maha tsara azy.",
    "Mety ho tsara kokoa raha toa ka...",
    "Tokony hojerena ny...",
    "Misy toerana azo anatsarana.",
    "Mety ho an'ny olona sasany ity vokatra ity.",
    "Tsy misy zavatra ratsy, fa tsy misy zavatra tsara be koa.",
    "Vokatra mahazatra, tsy misy miavaka.",
    "Mbola tsy nanandrana ny fampiasa rehetra aho.",
    "Mety mila fotoana vitsivitsy aho vao afaka manome hevitra feno.",
    "Miankina amin'ny filan'ny tsirairay ny maha tsara azy.",
    "Mety ho tsara kokoa raha toa ka...",
    "Tokony hojerena ny...",
    "Misy toerana azo anatsarana.",
    "Tsy dia nisy fiovana be tamin'ny vokatra teo aloha.",
    "Mety ho ampy ho an'ny fampiasana tsotra.",
    "Mila fanatsarana kely amin'ny antsipiriany.",
    "Tsy dia nahasarika ahy loatra ny famolavolana azy.",
    "Mety ho tsara kokoa ny kalitaon'ny fitaovana.",
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
num_comments = 10000
comments_data = generate_comments(num_comments)

# Convertir en DataFrame et sauvegarder en CSV
df = pd.DataFrame(comments_data, columns=["Commentaire", "Label"])
df.to_csv("src2/data/dataset.csv", index=False, encoding='utf-8')

print("Fichier CSV créé avec succès : dataset.csv")
