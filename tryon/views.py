# tryon/views.py
from pathlib import Path

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


def home(request):
    """Display the home page with try-on form."""
    return render(request, 'tryon/home.html', {'form': TryOnForm()})


def result(request, pk):
    """Display the try-on result."""
    obj = get_object_or_404(TryOnRequest, pk=pk)
    return render(request, 'tryon/result.html', {'item': obj})  # template expects 'item'


@require_POST
def submit_tryon(request):
    form = TryOnForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, "Please fix the errors in the form.")
        return render(request, 'tryon/home.html', {'form': form})

    # Determine mode (radio field): 'cloth' or 'prompt'
    mode = form.cleaned_data.get('mode', 'cloth')
    prompt = (form.cleaned_data.get('prompt') or "").strip()
    cloth_file = request.FILES.get('cloth_image')

    # Server-side validation for clarity
    if mode == 'cloth' and not cloth_file:
        messages.error(request, "Please upload a cloth image (or switch to 'Generate cloth from text').")
        return render(request, 'tryon/home.html', {'form': form})
    if mode == 'prompt' and not prompt:
        messages.error(request, "Please enter a clothing description (or switch to 'Upload a cloth image').")
        return render(request, 'tryon/home.html', {'form': form})

    # Save request (we set the flag from mode)
    obj: TryOnRequest = form.save(commit=False)
    obj.generate_from_prompt = (mode == 'prompt')
    obj.save()  # saves user_image (and cloth/mask if provided)

    # Output directory for intermediates & result
    out_dir = Path(settings.MEDIA_ROOT) / "outputs" / f"{obj.id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pose + body mask for the user image
    mask_path, pose_json_path = process_user_image(obj.user_image.path, out_dir)
    obj.body_mask.name = str(mask_path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")
    obj.pose_json.name = str(pose_json_path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")

    # 2) Resolve cloth source
    if obj.generate_from_prompt:
        # Generate cloth from text (fast settings handled inside the service)
        cloth_img_path, cloth_mask_path = generate_cloth_from_prompt(prompt, out_dir)
    else:
        # Use uploaded cloth; ensure there is a mask (auto-create if missing)
        cloth_img_path = Path(obj.cloth_image.path)
        if obj.cloth_mask:
            cloth_mask_path = Path(obj.cloth_mask.path)
        else:
            cloth_mask_path = auto_cloth_mask(cloth_img_path, out_dir / "auto_cloth_mask.png")

    # 3) Baseline fit / composite
    result_path = out_dir / "result.jpg"
    baseline_fit(
        user_img_path=Path(obj.user_image.path),
        pose_json_path=pose_json_path,
        cloth_img_path=Path(cloth_img_path),
        cloth_mask_path=Path(cloth_mask_path) if cloth_mask_path else None,
        out_path=result_path,
    )

    # Save result into the model
    obj.result_image.name = str(result_path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")
    obj.save(update_fields=["body_mask", "pose_json", "result_image", "generate_from_prompt"])

    # Helpful debug (shows in your server console)
    print("---- TRY-ON DEBUG ----")
    print("mode:", mode)
    print("user:", obj.user_image.path)
    print("mask:", obj.body_mask.path if obj.body_mask else None)
    print("pose:", obj.pose_json.path if obj.pose_json else None)
    print("cloth:", cloth_img_path)
    print("c_mask:", cloth_mask_path)
    print("result exists?", result_path.exists())
    print("----------------------")

    return redirect('result', pk=obj.pk)
