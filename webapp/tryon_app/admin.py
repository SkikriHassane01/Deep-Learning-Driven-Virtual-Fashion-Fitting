"""
Simple admin interface for try-on results.
"""

from django.contrib import admin
from .models import TryOnResult


@admin.register(TryOnResult)
class TryOnResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'clothing_prompt', 'processing_status', 'created_at']
    list_filter = ['processing_status', 'created_at']
    search_fields = ['clothing_prompt']
    readonly_fields = ['created_at']