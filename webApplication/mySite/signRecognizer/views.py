from django.shortcuts import render

# Create your views here.

def signRecognizer (request):
    return render(request,'index.html', None)
