import requests
from bs4 import BeautifulSoup
from googletrans import Translator

# Lancer la récupération de commentaires depuis un site (exemple avec un produit Amazon)
url = "https://www.amazon.fr/dp/B08P2X8ZZ9"  # Remplacer par l'URL du produit

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Récupérer tous les commentaires
commentaires = soup.find_all('span', {'data-asin': 'review-text'})
comments_list = [commentaire.get_text(strip=True) for commentaire in commentaires]

# Traduire les commentaires en malgache
translator = Translator()
comments_translated = [translator.translate(comment, src='fr', dest='mg').text for comment in comments_list]

# Afficher les commentaires traduits
for comment in comments_translated:
    print(comment)
