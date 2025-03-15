from django.db import models
from django.contrib.auth.models import User
from .ml_model import tester_commentaire  # Import du modèle IA

class Produit(models.Model):
    nom = models.CharField(max_length=255)
    description = models.TextField()
    prix = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return self.nom

class Commentaire(models.Model):
    produit = models.ForeignKey(Produit, on_delete=models.CASCADE, related_name="commentaires")
    utilisateur = models.CharField(max_length=100)  # Utilisateur sous forme de chaîne
    commentaire = models.TextField()
    date_ajout = models.DateTimeField(auto_now_add=True)
    sentiment = models.CharField(max_length=10, choices=[('Positif', 'Positif'), ('Neutre', 'Neutre'), ('Négatif', 'Négatif')], default='Neutre')

    def save(self, *args, **kwargs):
        """Avant d'enregistrer, applique l'analyse de sentiment avec l'IA."""
        self.sentiment = tester_commentaire(self.commentaire)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.utilisateur} - {self.sentiment}"
