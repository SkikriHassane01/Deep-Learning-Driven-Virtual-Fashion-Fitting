"""
Simple function-based views for virtual try-on app.
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta
import os
import json
import logging
from pathlib import Path

from .models import TryOnResult
from .services import tryon_service

logger = logging.getLogger(__name__)


def cleanup_stuck_results():
    """Clean up results that have been processing for more than 10 minutes"""
    cutoff_time = timezone.now() - timedelta(minutes=10)
    stuck_results = TryOnResult.objects.filter(
        processing_status='processing',
        created_at__lt=cutoff_time
    )
    
    for result in stuck_results:
        result.processing_status = 'failed'
        result.error_message = 'Processing timeout - request may have been stuck'
        result.save()
        logger.warning(f"Marked stuck result {result.id} as failed after timeout")
    
    return stuck_results.count()


def index_view(request):
    """Simplified home page without upload functionality"""
    context = {
        'ai_available': tryon_service.is_available()
    }
    return render(request, 'index.html', context)


@login_required
def generate_view(request):
    """AI Virtual Try-On Generation page - login required"""
    # Clean up any stuck results before showing the page
    cleanup_stuck_results()
    
    context = {
        'ai_available': tryon_service.is_available()
    }
    return render(request, 'generate.html', context)


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
    
    # If result is still processing and the user came from progress, redirect back
    if (tryon_result.processing_status == 'processing' and 
        request.META.get('HTTP_REFERER') and 
        'progress' not in request.META.get('HTTP_REFERER', '')):
        # Only show processing page if result was just created
        from django.utils import timezone
        time_diff = (timezone.now() - tryon_result.created_at).total_seconds()
        if time_diff < 300:  # Less than 5 minutes old
            logger.info(f"Showing processing status for recent result {result_id}")
        else:
            logger.warning(f"Old processing result {result_id} may be stuck")
    
    context = {
        'result': tryon_result
    }
    return render(request, 'result.html', context)




# progress_stream function removed - now using simple polling in frontend instead


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
            
            # Start actual processing in background
            import threading
            processing_thread = threading.Thread(
                target=process_image_background,
                args=(tryon_result.id,)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
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


def process_image_background(result_id):
    """Background function to process the image"""
    try:
        tryon_result = TryOnResult.objects.get(id=result_id)
        
        # Process the image
        image_path = tryon_result.original_image.path
        result = tryon_service.process_image(image_path, tryon_result.clothing_prompt)
        
        if result['success']:
            # Update database with result
            result_relative_path = Path(result['result_path']).relative_to(
                Path(tryon_result.original_image.storage.location)
            )
            tryon_result.result_image = str(result_relative_path)
            tryon_result.processing_status = 'completed'
            tryon_result.save()
            logger.info(f"Processing completed successfully for result {result_id}")
        else:
            # Update database with error
            tryon_result.processing_status = 'failed'
            tryon_result.error_message = result['error']
            tryon_result.save()
            logger.error(f"Processing failed for result {result_id}: {result['error']}")
            
    except Exception as e:
        logger.error(f"Background processing error for result {result_id}: {e}")
        try:
            tryon_result = TryOnResult.objects.get(id=result_id)
            tryon_result.processing_status = 'failed'
            tryon_result.error_message = str(e)
            tryon_result.save()
        except:
            pass


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
    user_results = TryOnResult.objects.filter(
        user=request.user,
        processing_status='completed',
        result_image__isnull=False
    ).exclude(result_image='').order_by('-created_at')
    
    context = {
        'user_results': user_results
    }
    return render(request, 'registration/profile.html', context)