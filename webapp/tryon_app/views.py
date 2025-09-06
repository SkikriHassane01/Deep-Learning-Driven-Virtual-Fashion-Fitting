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
from django.utils import timezone
from datetime import timedelta
import os
import json
import time
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
    """Home page with upload form"""
    # Clean up any stuck results before showing the page
    cleanup_stuck_results()
    
    if request.user.is_authenticated:
        recent_results = TryOnResult.objects.filter(
            user=request.user, 
            processing_status='completed',
            result_image__isnull=False
        ).exclude(result_image='').order_by('-created_at')[:6]
    else:
        recent_results = TryOnResult.objects.filter(
            processing_status='completed',
            result_image__isnull=False
        ).exclude(result_image='').order_by('-created_at')[:6]
    
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




def progress_stream(request, result_id):
    """Stream processing progress updates"""
    def event_stream():
        # Get the result object
        try:
            tryon_result = TryOnResult.objects.get(id=result_id)
        except TryOnResult.DoesNotExist:
            yield f"data: {json.dumps({'error': 'Result not found'})}\n\n"
            return
        
        # Progress updates aligned with actual AI processing pipeline
        steps = [
            {"progress": 10, "text": "Loading and preprocessing your image...", "duration": 3.0},
            {"progress": 25, "text": "Running human pose detection...", "duration": 8.0},
            {"progress": 40, "text": "Segmenting body regions and clothing areas...", "duration": 12.0},
            {"progress": 55, "text": "Creating garment placement mask...", "duration": 6.0},
            {"progress": 95, "text": "AI diffusion model generating clothing", "duration": 120.0, "has_substeps": True},  # 2 minutes for 25 steps
            {"progress": 100, "text": "Post-processing and finalizing result...", "duration": 4.0},
        ]
        
        for i, phase in enumerate(steps):
            if phase.get('has_substeps'):  # AI generation phase with diffusion steps
                for diffusion_step in range(1, 26):
                    progress = 55 + ((diffusion_step / 25) * 40)  # 55% to 95%
                    step_msg = f"AI diffusion model generating clothing (Step {diffusion_step}/25)..."
                    data = {
                        'progress': int(progress),
                        'step': step_msg,
                        'status': 'processing',
                        'phase': i + 1,
                        'diffusion_step': diffusion_step
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(phase['duration'] / 25)  # Distribute time across 25 steps
            else:
                data = {
                    'progress': int(phase['progress']),
                    'step': phase['text'],
                    'status': 'processing',
                    'phase': i + 1
                }
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(phase['duration'])
        
        # After simulation, wait for actual processing to complete
        while True:
            # Refresh the tryon_result from database
            tryon_result.refresh_from_db()
            
            if tryon_result.processing_status == 'completed':
                data = {
                    'progress': 100,
                    'step': "Processing completed successfully!",
                    'status': 'completed',
                    'phase': 6
                }
                yield f"data: {json.dumps(data)}\n\n"
                break
            elif tryon_result.processing_status == 'failed':
                data = {
                    'progress': 100,
                    'step': f"Processing failed: {tryon_result.error_message}",
                    'status': 'error',
                    'error': tryon_result.error_message
                }
                yield f"data: {json.dumps(data)}\n\n"
                break
            else:
                # Still processing, wait a bit
                time.sleep(2)
                # Update progress text to show we're waiting
                data = {
                    'progress': 95,
                    'step': "Finalizing results, please wait...",
                    'status': 'processing',
                    'phase': 6
                }
                yield f"data: {json.dumps(data)}\n\n"
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['Connection'] = 'keep-alive'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response


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