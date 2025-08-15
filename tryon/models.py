from django.db import models

class TryOnRequest(models.Model):
    user_image = models.ImageField(upload_to='uploads/user/')
    prompt = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    # place to store outputs we’ll add later (pose, mask, try-on result)
    result_image = models.ImageField(upload_to='outputs/', null=True, blank=True)

    def __str__(self):
        return f"TryOnRequest #{self.id} - {self.prompt[:30]}"
