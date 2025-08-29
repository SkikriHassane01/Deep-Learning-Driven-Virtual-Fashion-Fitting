from django.db import models

class TryOnRequest(models.Model):
    user_image = models.ImageField(upload_to='uploads/user/')
    prompt = models.CharField(max_length=255, default="Default clothing description")  # Added default
    # Store path to final image (relative to MEDIA_ROOT)
    result_image = models.ImageField(upload_to='results/', blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"TryOnRequest #{self.pk} - {self.prompt[:40]}"