from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views import View
from django.views.generic.edit import FormView
from classifier.gm import predict
from .forms import ClassifierForm
# Create your views here.

class ClassifierView(View):
    
    def get(self,request):
        form = ClassifierForm()
        return render(request, "classifier/index.html", {"form":form})
    
    def post(self,request):
        print("I am here")
        form = ClassifierForm(request.POST, request.FILES)
        if form.is_valid():
            print(form)
            form.save()
            cls = form.instance
            print(cls.input_image.path)
            pred = predict.predict(cls.input_image.path)
            print("here like 24")
            print(pred)
            return render(request,"classifier/index.html", {"form":form, "cls" : cls,'pred': pred})
        
        return render("classifier/index.html", {"form":form})
        
        
    