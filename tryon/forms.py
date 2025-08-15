from django import forms
from .models import TryOnRequest

class TryOnForm(forms.ModelForm):
    generate_from_prompt = forms.BooleanField(
        required=False, initial=True,  # default ON
        help_text="Generate a cloth from your text instead of uploading one."
    )

    class Meta:
        model = TryOnRequest
        fields = ['user_image', 'cloth_image', 'cloth_mask', 'prompt', 'generate_from_prompt']
        widgets = {
            'prompt': forms.TextInput(attrs={'placeholder': 'e.g. "a red dress with floral patterns"'}),
        }
