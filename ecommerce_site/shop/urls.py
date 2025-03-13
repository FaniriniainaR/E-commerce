from django.urls import path
from . import views

urlpatterns = [
    path('', views.liste_produits, name='liste_produits'),
    path('produit/<int:produit_id>/', views.produit_detail, name='produit_detail'),
    path('paiement/', views.paiement_simule, name='paiement_simule'),
]
