from django.urls import path
from . import views

urlpatterns = [
    path('', views.liste_produits, name='liste_produits'),
    path('produit/<int:produit_id>/', views.produit_detail, name='produit_detail'),
    path('ajouter_au_panier/<int:produit_id>/', views.ajouter_au_panier, name='ajouter_au_panier'),
    path('panier/', views.panier, name='panier'),
    path('paiement/', views.paiement_simule, name='paiement_simule'),
    path('paiement/<int:produit_id>/', views.paiement_simule, name='paiement_simule')

]
