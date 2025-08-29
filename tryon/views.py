# tryon/views.py
import os
from pathlib import Path
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods, require_POST

from .forms import TryOnForm
from .models import TryOnRequest
from .services.pose_seg import process_user_image, PoseSegError
from .services.text2cloth import generate_cloth_from_prompt
from .services.auto_mask import auto_cloth_mask, ClothMaskError
from .services.fitter import baseline_fit, TryOnError

@require_http_methods(["GET"])
def home(request):
    form = TryOnForm()
    return render(request, "tryon/home.html", {"form": form})

@require_POST
def submit_tryon(request):
    form = TryOnForm(request.POST, request.FILES)
    if not form.is_valid():
        for f, errs in form.errors.items():
            for e in errs:
                messages.error(request, f"{f}: {e}")
        return redirect("home")

    obj = form.save()
    req_id = obj.pk

    media_root = Path(settings.MEDIA_ROOT)
    out_root = media_root / "outputs" / str(req_id)
    pose_dir = out_root / "pose"
    gen_dir  = out_root / "generated"
    mask_dir = out_root / "masks"
    for d in (pose_dir, gen_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    person_img = Path(obj.user_image.path)

    try:
        # 1) Pose + segmentation (strict)
        pose_json_path, _ = process_user_image(person_img, pose_dir)

        # 2) Text → cloth (strict alpha)
        cloth_img_path, _ = generate_cloth_from_prompt(obj.prompt, gen_dir)

        # 3) Cloth mask (strict alpha-only)
        cloth_mask_path = mask_dir / "cloth_mask.png"
        auto_cloth_mask(cloth_img_path, cloth_mask_path)

        # 4) Try-on (model-only)
        result_path = baseline_fit(
            person_image_path=str(person_img),
            cloth_image_path=str(cloth_img_path),
            pose_json_path=str(pose_json_path),
            cloth_mask_path=str(cloth_mask_path),
        )

        # Fix relative path calculation
        result_path_obj = Path(result_path)
        if result_path_obj.is_absolute():
            try:
                rel = result_path_obj.relative_to(media_root).as_posix()
            except ValueError:
                # If path is not relative to media_root, just use the filename
                rel = f"results/{result_path_obj.name}"
        else:
            rel = result_path
        
        obj.result_image = rel
        obj.save()
        messages.success(request, "Try-on completed!")
    except (PoseSegError, ClothMaskError, TryOnError) as e:
        messages.error(request, f"Try-on failed: {e}")
    except Exception as e:
        messages.error(request, f"Unexpected error: {e}")

    return redirect("result", pk=req_id)

@require_http_methods(["GET"])
def result_view(request, pk: int):
    obj = get_object_or_404(TryOnRequest, pk=pk)
    return render(request, "tryon/result.html", {"obj": obj})
