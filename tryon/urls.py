from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),               
    path('submit/', views.submit_tryon, name='submit'),
    path('result/<int:pk>/', views.result, name='result'),
    path('debug/', views.debug_view, name='debug'),  # Add a debug route
]
