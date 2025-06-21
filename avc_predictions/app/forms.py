from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Utilisateur,RawDataset

class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True)
    username = forms.CharField(max_length=150, required=True)
    profile_picture_path = forms.FileField(required=True)
    country = forms.CharField(max_length=100, required=True)
    full_name = forms.CharField(max_length=150, required=True)
    STATUS_CHOICES = [
        ('Personal', 'Personal'),
        ('Organisation (hospital/clinique)', 'Organisation (hospital/clinique)'),
    ]
    status = forms.ChoiceField(choices=STATUS_CHOICES, required=True, widget=forms.Select(attrs={'class': 'form-control'}))

    class Meta:
        model = Utilisateur
        fields = ['username', 'email', 'password1', 'password2', 'profile_picture_path', 'country', 'full_name', 'status']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'password1': forms.PasswordInput(attrs={'class': 'form-control'}),
            'password2': forms.PasswordInput(attrs={'class': 'form-control'}),
            'profile_picture_path': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'country': forms.TextInput(attrs={'class': 'form-control'}),
            'full_name': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.profile_picture_path = self.cleaned_data['profile_picture_path']
        user.country = self.cleaned_data['country']
        user.full_name = self.cleaned_data['full_name']
        user.status = self.cleaned_data.get('status', '')  # Ensure empty status works
        if commit:
            user.save()
        return user

class RawForm(forms.ModelForm):
    class Meta:
        model = RawDataset
        fields = ["datasetCostumName", "file_raw_dataset","TargetColumn"]
    