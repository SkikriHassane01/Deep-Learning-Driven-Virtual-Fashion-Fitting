# tryon/views.py
from pathlib import Path
import os
import sys
import django

from django.conf import settings
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import TryOnForm
from .models import TryOnRequest
from .services.auto_mask import auto_cloth_mask
from .services.fitter import baseline_fit
from .services.pose_seg import process_user_image
from .services.text2cloth import generate_cloth_from_prompt

try:
    import torch
except ImportError:
    torch = None


def home(request):
    return render(request, 'tryon/home.html', {'form': TryOnForm()})


def result(request, pk):
    obj = get_object_or_404(TryOnRequest, pk=pk)
    return render(request, 'tryon/result.html', {'item': obj})


def debug_view(request):
    debug_info = {
        'python_version': sys.version,
        'django_version': django.get_version(),
        'cuda_available': (torch.cuda.is_available() if torch else 'torch not imported'),
        'media_root': settings.MEDIA_ROOT,
        'base_dir': settings.BASE_DIR,
        'recent_requests': TryOnRequest.objects.all().order_by('-created_at')[:5],
    }
    return render(request, 'tryon/debug.html', {'debug_info': debug_info})


@require_POST
def submit_tryon(request):
    form = TryOnForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, "Please check your form inputs.")
        return redirect('home')

    obj = form.save()
    request_id = obj.id
    print(f"Processing try-on request {request_id}")

    # --- Per-request output folders ---
    media_root = Path(settings.MEDIA_ROOT)
    out_root   = media_root / 'outputs' / str(request_id)
    pose_dir   = out_root / 'pose'
    mask_dir   = out_root / 'masks'
    gen_dir    = out_root / 'generated'
    result_dir = media_root / 'results'
    for d in (pose_dir, mask_dir, gen_dir, result_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Resolve inputs as Path objects ---
    person_image_path = Path(obj.user_image.path)
    cloth_image_path = Path(obj.cloth_image.path) if obj.cloth_image else None
    cloth_mask_path = None
    pose_json_path = None

    try:
        # A) Cloth: upload vs text prompt
        if cloth_image_path:
            print(f"Upload mode: Using cloth image {cloth_image_path}")
        elif obj.text_prompt:
            try:
                # generate_cloth_from_prompt(prompt, output_dir) -> (cloth_img_path, cloth_mask_path|None)
                cloth_img_str, cloth_mask_str = generate_cloth_from_prompt(obj.text_prompt, str(gen_dir))
                cloth_image_path = Path(cloth_img_str)
                cloth_mask_path = Path(cloth_mask_str) if cloth_mask_str else None
                print(f"Text mode: Generated cloth {cloth_image_path}")
            except Exception as gen_error:
                print(f"Error generating cloth from prompt: {gen_error}")
                messages.error(request, f"Failed to generate clothing from prompt: {gen_error}")
                return redirect('result', pk=request_id)
        else:
            messages.error(request, "No cloth image or text prompt provided.")
            return redirect('result', pk=request_id)

        # B) Pose JSON: expected path under pose_dir
        expected_pose_json = pose_dir / f"{person_image_path.stem}_pose.json"
        print(f"Person image: {person_image_path}")
        print(f"Pose JSON (expected): {expected_pose_json}")
        print(f"Pose exists: {expected_pose_json.exists()}")

        if expected_pose_json.exists():
            pose_json_path = expected_pose_json
        else:
            try:
                print("Generating pose data...")
                # PASS Path objects (not strings)
                process_user_image(person_image_path, pose_dir)
                pose_json_path = expected_pose_json
                print(f"Pose data generated: {pose_json_path.exists()} at {pose_json_path}")
            except Exception as pose_error:
                print(f"Failed to generate pose data: {pose_error}")
                pose_json_path = None  # proceed without pose

        if not (pose_json_path and pose_json_path.exists()):
            print("No pose data available - this may result in poor quality")

        # C) Cloth mask: if absent, auto-generate into mask_dir
        if cloth_image_path and not cloth_mask_path:
            try:
                print("Generating cloth mask...")
                target_mask = mask_dir / (cloth_image_path.stem + '_mask.png')
                # PASS Path objects (not strings)
                ret_mask = auto_cloth_mask(cloth_image_path, target_mask)
                cloth_mask_path = Path(ret_mask) if ret_mask else target_mask
                print(f"Cloth mask generated: {cloth_mask_path} (exists={cloth_mask_path.exists()})")
            except Exception as mask_error:
                print(f"Failed to generate cloth mask: {mask_error}")
                cloth_mask_path = None

        if not (cloth_mask_path and cloth_mask_path.exists()):
            print("No cloth mask available - this may result in poor quality")

        # D) Try-on (baseline_fit uses model first, then fallback if needed)
        try:
            result_path = baseline_fit(
                person_image_path=str(person_image_path),
                cloth_image_path=str(cloth_image_path),
                pose_json_path=str(pose_json_path) if (pose_json_path and pose_json_path.exists()) else None,
                cloth_mask_path=str(cloth_mask_path) if (cloth_mask_path and cloth_mask_path.exists()) else None,
            )

            print(f"Try-on completed. Result saved to: {result_path}")
            result_path = Path(result_path)
            print(f"Result file exists: {result_path.exists()}")

            if result_path.exists():
                relative_path = os.path.relpath(str(result_path), settings.MEDIA_ROOT).replace('\\', '/')
                obj.result_image = relative_path
                obj.save()
                messages.success(request, "Try-on completed successfully!")
                print(f"Result image saved as: {relative_path}")
            else:
                raise FileNotFoundError(f"Result file not found at {result_path}")

        except Exception as tryon_error:
            print(f"Try-on process failed: {tryon_error}")
            messages.error(request, f"Try-on failed: {tryon_error}")

    except Exception as e:
        print(f"Error in submit_tryon: {e}")
        messages.error(request, f"An error occurred: {e}")

    return redirect('result', pk=request_id)


def virtual_tryon(request):
    if request.method == 'POST':
        form = TryOnForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            relative_path = f'outputs/{obj.id}/result_image.jpg'
            return render(request, 'tryon/result.html', {'processed_image': relative_path})
    return render(request, 'tryon/upload.html')
