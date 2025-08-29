# tryon/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("submit/", views.submit_tryon, name="submit"),
    path("result/<int:pk>/", views.result_view, name="result"),
]
