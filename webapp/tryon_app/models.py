"""
Simple database models for virtual try-on app.
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User


class TryOnResult(models.Model):
    """Simple model to store try-on results"""
    
    # User association (optional for backward compatibility)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, help_text='User who created this try-on')
    
    # User input
    original_image = models.ImageField(upload_to='uploads/', help_text='Original user photo')
    clothing_prompt = models.CharField(max_length=500, help_text='Text description of desired clothing')
    
    # Generated results
    result_image = models.ImageField(upload_to='results/', blank=True, null=True, help_text='Final try-on result')
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
        ],
        default='pending'
    )
    error_message = models.TextField(blank=True, help_text='Error details if processing failed')
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Try-On Result'
        verbose_name_plural = 'Try-On Results'
    
    def __str__(self):
        return f"Try-on {self.id}: {self.clothing_prompt[:50]}..."