from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),   # home
    path('predict/', views.predict, name='predict'),  # video prediction
]