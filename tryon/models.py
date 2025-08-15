from django.db import models

class TryOnRequest(models.Model):
    user_image = models.ImageField(upload_to='uploads/user/')
    cloth_image = models.ImageField(upload_to='uploads/cloth/', null=True, blank=True) 
    cloth_mask  = models.ImageField(upload_to='uploads/cloth_mask/', null=True, blank=True)
    
    prompt = models.CharField(max_length=255)

    generate_from_prompt = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    # analysis outputs
    body_mask = models.ImageField(upload_to='outputs/masks/', null=True, blank=True)
    pose_json = models.FileField(upload_to='outputs/pose/', null=True, blank=True)

    # final try-on output
    result_image = models.ImageField(upload_to='outputs/', null=True, blank=True)

    def __str__(self):
        return f"TryOnRequest #{self.id} - {self.prompt[:30]}"