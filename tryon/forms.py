from django import forms
from .models import TryOnRequest

class TryOnForm(forms.ModelForm):
    class Meta:
        model = TryOnRequest
        fields = ['user_image', 'prompt']
        widgets = {
            'prompt': forms.TextInput(attrs={'placeholder': 'e.g. "red floral dress"'}),
        }
