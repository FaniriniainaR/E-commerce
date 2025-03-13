from django.db import models

class Produit(models.Model):
    nom = models.CharField(max_length=200)
    description = models.TextField()
    prix = models.DecimalField(max_digits=10, decimal_places=2)
    image = models.ImageField(upload_to='produits/')

    def __str__(self):
        return self.nom

class Commentaire(models.Model):
    produit = models.ForeignKey(Produit, on_delete=models.CASCADE)
    utilisateur = models.CharField(max_length=100)
    commentaire = models.TextField()
    date_ajout = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Commentaire sur {self.produit.nom} par {self.utilisateur}"

