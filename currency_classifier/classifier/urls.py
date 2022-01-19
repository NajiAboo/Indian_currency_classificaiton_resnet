
from django.urls import path
from .views import ClassifierView

urlpatterns = [
    path("",ClassifierView.as_view()),
]
