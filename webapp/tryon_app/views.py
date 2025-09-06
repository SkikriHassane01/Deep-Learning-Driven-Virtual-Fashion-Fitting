"""
Simple function-based views for virtual try-on app.
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.urls import reverse
import os
import json
import time
from pathlib import Path

from .models import TryOnResult
from .services import tryon_service


def index_view(request):
    """Home page with upload form"""
    if request.user.is_authenticated:
        recent_results = TryOnResult.objects.filter(
            user=request.user, 
            processing_status='completed'
        ).order_by('-created_at')[:6]
    else:
        recent_results = TryOnResult.objects.filter(processing_status='completed')[:6]
    
    context = {
        'recent_results': recent_results,
        'ai_available': tryon_service.is_available()
    }
    return render(request, 'index.html', context)


@login_required
def upload_view(request):
    """Handle image upload and text input"""
    if request.method == 'POST':
        # Get form data
        image_file = request.FILES.get('image')
        clothing_prompt = request.POST.get('prompt', '').strip()
        
        # Simple validation
        if not image_file:
            messages.error(request, 'Please select an image to upload.')
            return redirect('index')
        
        if not clothing_prompt:
            messages.error(request, 'Please enter a description of the desired clothing.')
            return redirect('index')
        
        try:
            # Create database record
            tryon_result = TryOnResult.objects.create(
                original_image=image_file,
                clothing_prompt=clothing_prompt,
                processing_status='processing',
                user=request.user if request.user.is_authenticated else None
            )
            
            # Process the image
            image_path = tryon_result.original_image.path
            result = tryon_service.process_image(image_path, clothing_prompt)
            
            if result['success']:
                # Update database with result
                result_relative_path = Path(result['result_path']).relative_to(
                    Path(tryon_result.original_image.storage.location)
                )
                tryon_result.result_image = str(result_relative_path)
                tryon_result.processing_status = 'completed'
                tryon_result.save()
                
                messages.success(request, result['message'])
                return redirect('result', result_id=tryon_result.id)
            else:
                # Update database with error
                tryon_result.processing_status = 'failed'
                tryon_result.error_message = result['error']
                tryon_result.save()
                
                messages.error(request, f"Processing failed: {result['error']}")
                return redirect('index')
                
        except Exception as e:
            messages.error(request, f'Upload failed: {str(e)}')
            return redirect('index')
    
    # GET request - redirect to home
    return redirect('index')


def result_view(request, result_id):
    """Display try-on result"""
    tryon_result = get_object_or_404(TryOnResult, id=result_id)
    
    context = {
        'result': tryon_result
    }
    return render(request, 'result.html', context)


@login_required
def gallery_view(request):
    """User-specific gallery of try-on results"""
    results = TryOnResult.objects.filter(
        user=request.user,
        processing_status='completed'
    ).order_by('-created_at')
    
    context = {
        'results': results
    }
    return render(request, 'gallery.html', context)


def progress_stream(request, result_id):
    """Stream processing progress updates"""
    def event_stream():
        # Get the result object
        try:
            tryon_result = TryOnResult.objects.get(id=result_id)
        except TryOnResult.DoesNotExist:
            yield f"data: {json.dumps({'error': 'Result not found'})}\n\n"
            return
        
        # Simulate progress updates
        steps = [
            "Initializing AI pipeline...",
            "Loading your image...",
            "Analyzing human pose...",
            "Creating clothing mask...",
            "Generating new clothing...",
            "Processing diffusion (Step 1/25)...",
        ]
        
        for i, step in enumerate(steps):
            if i == 5:  # Start diffusion steps
                for diffusion_step in range(1, 26):
                    progress = 20 + (diffusion_step / 25) * 60  # 20% to 80%
                    step_msg = f"Generating clothing (Step {diffusion_step}/25)..."
                    data = {
                        'progress': int(progress),
                        'step': step_msg,
                        'status': 'processing'
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(0.2)  # Simulate processing time
            else:
                progress = (i + 1) * 15  # Rough progress calculation
                data = {
                    'progress': min(progress, 80),
                    'step': step,
                    'status': 'processing'
                }
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(0.5)
        
        # Final step
        data = {
            'progress': 100,
            'step': "Finalizing result...",
            'status': 'completed'
        }
        yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


@login_required
@csrf_exempt
def upload_async(request):
    """Async upload endpoint for better progress tracking"""
    if request.method == 'POST':
        # Get form data
        image_file = request.FILES.get('image')
        clothing_prompt = request.POST.get('prompt', '').strip()
        
        # Simple validation
        if not image_file or not clothing_prompt:
            return JsonResponse({
                'success': False,
                'error': 'Missing image or prompt'
            })
        
        try:
            # Create database record
            tryon_result = TryOnResult.objects.create(
                original_image=image_file,
                clothing_prompt=clothing_prompt,
                processing_status='processing',
                user=request.user if request.user.is_authenticated else None
            )
            
            # Return the result ID for progress tracking
            return JsonResponse({
                'success': True,
                'result_id': tryon_result.id,
                'message': 'Upload successful, processing started'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


# Authentication Views


def register_view(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            login(request, user)
            messages.success(request, f'Welcome {username}! Your account has been created successfully.')
            return redirect('index')
    else:
        form = UserCreationForm()
    
    context = {'form': form}
    return render(request, 'registration/register.html', context)


@login_required
def profile_view(request):
    """User profile view showing their try-on history"""
    user_results = TryOnResult.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'user_results': user_results
    }
    return render(request, 'registration/profile.html', context)