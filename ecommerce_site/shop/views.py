from django.shortcuts import render, redirect
from .models import Produit, Commentaire
from .forms import CommentaireForm

def liste_produits(request):
    produits = Produit.objects.all()
    # Ajouter le panier à la vue pour l'affichage du compteur
    panier = request.session.get('panier', {})
    return render(request, 'shop/liste_produits.html', {
        'produits': produits,
        'panier': panier
    })

def produit_detail(request, produit_id):
    produit = Produit.objects.get(id=produit_id)
    commentaires = Commentaire.objects.filter(produit=produit)

    # Vérifier si l'utilisateur a effectué le paiement
    if not request.session.get('a_paye', False):
        return redirect('paiement_simule')

    # Ajouter un commentaire si le formulaire est soumis
    if request.method == 'POST':
        form = CommentaireForm(request.POST)
        if form.is_valid():
            commentaire = form.save(commit=False)
            commentaire.produit = produit
            commentaire.utilisateur = request.user.username if request.user.is_authenticated else "Anonyme"
            commentaire.save()
            return redirect('produit_detail', produit_id=produit.id)
    else:
        form = CommentaireForm()

    return render(request, 'shop/produit_detail.html', {
        'produit': produit,
        'commentaires': commentaires,
        'form': form
    })

def paiement_simule(request, produit_id=None):
    panier = request.session.get('panier', {})
    
    if not panier:
        return redirect('panier')
    
    # Marquer le paiement comme effectué
    request.session['a_paye'] = True
    
    # Vider le panier après paiement
    request.session['panier'] = {}
    
    # Si un produit_id est fourni, rediriger vers ce produit
    if produit_id:
        return redirect('produit_detail', produit_id=produit_id)
    
    return redirect('liste_produits')




# Vue pour ajouter un produit au panier
def ajouter_au_panier(request, produit_id):
    produit = Produit.objects.get(id=produit_id)
    
    # Récupérer le panier dans la session, s'il existe
    panier = request.session.get('panier', {})
    
    # Ajouter le produit au panier (ou mettre à jour la quantité si le produit est déjà dans le panier)
    if str(produit.id) in panier:
        panier[str(produit.id)]['quantite'] += 1
    else:
        panier[str(produit.id)] = {
            'nom': produit.nom,
            'prix': str(produit.prix),
            'quantite': 1,
        }
    
    # Sauvegarder le panier dans la session
    request.session['panier'] = panier
    
    return redirect('panier')

# Vue pour afficher le panier
def panier(request):
    panier = request.session.get('panier', {})
    total = 0
    for produit_id, details in panier.items():
        total += float(details['prix']) * details['quantite']
    
    return render(request, 'shop/panier.html', {
        'panier': panier,
        'total': total,
    })
