from django.shortcuts import render, redirect, get_object_or_404
from .forms import TryOnForm
from .models import TryOnRequest

def home(request):
    form = TryOnForm()
    return render(request, 'tryon/home.html', {'form': form})

def submit_tryon(request):
    if request.method == 'POST':
        form = TryOnForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()  # saves the uploaded image to MEDIA_ROOT
            # Later we’ll trigger pose/segmentation + generation here, then save result_image
            return redirect('result', pk=obj.pk)
        else:
            return render(request, 'tryon/home.html', {'form': form})
    return redirect('home')

def result(request, pk):
    obj = get_object_or_404(TryOnRequest, pk=pk)
    return render(request, 'tryon/result.html', {'item': obj})
