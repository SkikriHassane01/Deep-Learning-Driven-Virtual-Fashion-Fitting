from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from pathlib import Path

from .forms import TryOnForm
from .models import TryOnRequest
from .services.pose_seg import process_user_image
from .services.fitter import baseline_fit
from .services.text2cloth import generate_cloth_from_prompt
from .services.auto_mask import auto_cloth_mask

def home(request):
    """Display the home page with try-on form."""
    form = TryOnForm()
    return render(request, 'tryon/home.html', {'form': form})

def result(request, pk):
    """Display the try-on result."""
    obj = get_object_or_404(TryOnRequest, pk=pk)
    return render(request, 'tryon/result.html', {'tryon_request': obj})

def submit_tryon(request):
    if request.method != 'POST':
        return redirect('home')

    form = TryOnForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, 'tryon/home.html', {'form': form})

    obj: TryOnRequest = form.save(commit=False)
    gen_flag = form.cleaned_data.get('generate_from_prompt', False)
    obj.generate_from_prompt = bool(gen_flag)
    obj.save()  # saves user_image (and cloth if provided)

    out_dir = Path(settings.MEDIA_ROOT) / "outputs" / f"{obj.id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pose + body mask for the user image
    mask_path, pose_json_path = process_user_image(obj.user_image.path, out_dir)
    obj.body_mask.name = str(mask_path.relative_to(settings.MEDIA_ROOT)).replace("\\","/")
    obj.pose_json.name = str(pose_json_path.relative_to(settings.MEDIA_ROOT)).replace("\\","/")

    # 2) Decide cloth source
    cloth_img_path = None
    cloth_mask_path = None

    if obj.generate_from_prompt or not obj.cloth_image:
        # Generate from text
        cloth_img_path, cloth_mask_path = generate_cloth_from_prompt(obj.prompt, out_dir)
    else:
        # Use uploaded cloth; ensure mask exists (auto if missing)
        cloth_img_path = Path(obj.cloth_image.path)
        if obj.cloth_mask:
            cloth_mask_path = Path(obj.cloth_mask.path)
        else:
            cloth_mask_path = auto_cloth_mask(cloth_img_path, out_dir / "auto_cloth_mask.png")

    # 3) Baseline fit
    result_path = out_dir / "result.jpg"
    baseline_fit(
        user_img_path=Path(obj.user_image.path),
        pose_json_path=pose_json_path,
        cloth_img_path=Path(cloth_img_path),
        cloth_mask_path=Path(cloth_mask_path) if cloth_mask_path else None,
        out_path=result_path
    )
    obj.result_image.name = str(result_path.relative_to(settings.MEDIA_ROOT)).replace("\\","/")
    obj.save(update_fields=["body_mask","pose_json","result_image","generate_from_prompt"])

    return redirect('result', pk=obj.pk)
