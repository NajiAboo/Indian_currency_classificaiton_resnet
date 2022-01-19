from django.db import models

# Create your models here.

class Classifier(models.Model):
    input_image = models.ImageField()
    output_image = models.ImageField()
    