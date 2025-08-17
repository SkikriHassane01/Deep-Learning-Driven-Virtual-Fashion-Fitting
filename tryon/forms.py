from django import forms
from .models import TryOnRequest

class TryOnForm(forms.ModelForm):
    MODE_CHOICES = [
        ("cloth", "Upload a cloth image"),
        ("prompt", "Generate cloth from text"),
    ]
    mode = forms.ChoiceField(choices=MODE_CHOICES, initial="cloth", widget=forms.RadioSelect, required=True)

    class Meta:
        model = TryOnRequest
        fields = ['user_image', 'cloth_image', 'prompt']
        widgets = {
            'prompt': forms.TextInput(attrs={
                'placeholder': 'e.g. "red floral dress with floral patterns"',
                'required': False
            }),
            'user_image': forms.FileInput(attrs={'required': True}),
            'cloth_image': forms.FileInput(attrs={'required': False}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make prompt field not required by default
        self.fields['prompt'].required = False
        self.fields['cloth_image'].required = False
    
    def clean(self):
        cleaned_data = super().clean()
        mode = cleaned_data.get('mode')
        prompt = cleaned_data.get('prompt')
        cloth_image = cleaned_data.get('cloth_image')
        
        if mode == 'prompt' and not prompt:
            self.add_error('prompt', 'Please provide a text description when generating from prompt')
        
        if mode == 'cloth' and not cloth_image:
            self.add_error('cloth_image', 'Please upload a cloth image')
            
        return cleaned_data
