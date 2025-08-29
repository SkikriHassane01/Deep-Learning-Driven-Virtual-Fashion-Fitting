# tryon/forms.py
from django import forms
from .models import TryOnRequest

class TryOnForm(forms.ModelForm):
    class Meta:
        model = TryOnRequest
        fields = ['user_image', 'prompt']
        widgets = {
            'prompt': forms.TextInput(attrs={
                'placeholder': 'Describe the outfit (e.g. "red floral summer dress")',
            }),
            'user_image': forms.FileInput(),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['prompt'].required = True
        self.fields['user_image'].required = True

    def clean(self):
        cleaned = super().clean()
        prompt = cleaned.get('prompt', '')
        if not prompt.strip():
            self.add_error('prompt', 'Please provide a description.')
        return cleaned
