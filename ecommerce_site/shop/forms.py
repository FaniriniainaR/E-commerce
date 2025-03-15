from django import forms
from .models import Commentaire

class CommentaireForm(forms.ModelForm):
    class Meta:
        model = Commentaire
        fields = ['commentaire']

    def clean_commentaire(self):
        """Validation pour vérifier que le commentaire n'est pas vide"""
        commentaire = self.cleaned_data.get('commentaire')
        if not commentaire:
            raise forms.ValidationError("Le commentaire ne peut pas être vide.")
        return commentaire
