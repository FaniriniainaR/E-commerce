from django.shortcuts import render, redirect
from .models import Produit, Commentaire
from .forms import CommentaireForm

# Vue pour afficher la liste des produits
def liste_produits(request):
    produits = Produit.objects.all()
    return render(request, 'shop/liste_produits.html', {'produits': produits})

# Vue pour afficher un produit avec les commentaires et permettre l'ajout
def produit_detail(request, produit_id):
    produit = Produit.objects.get(id=produit_id)
    commentaires = Commentaire.objects.filter(produit=produit)

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

# Vue pour simuler un paiement (simple)
def paiement_simule(request):
    return render(request, 'shop/paiement_simule.html')
