
from django import forms
from .models import Classifier

class ClassifierForm(forms.ModelForm):
    class Meta:
        model = Classifier
        fields = ["input_image"]
        labels = {
            "input_image" : "Please Select the Indian Currency"
        }
        
         