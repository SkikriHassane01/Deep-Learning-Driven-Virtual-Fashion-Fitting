"""
URL configuration for tryon_app.
"""
from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Main app URLs
    path('', views.index_view, name='index'),
    path('upload/', views.upload_view, name='upload'),
    path('result/<int:result_id>/', views.result_view, name='result'),
    path('gallery/', views.gallery_view, name='gallery'),
    
    # Authentication URLs
    path('register/', views.register_view, name='register'),
    path('profile/', views.profile_view, name='profile'),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)