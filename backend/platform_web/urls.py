from django.contrib import admin
from django.urls import include, path
from django.contrib.auth.views import LoginView, LogoutView

from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("orchestration.urls")),
    path("", views.home, name="home"),
    path("accounts/login/", LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("accounts/logout/", LogoutView.as_view(next_page="home"), name="logout"),
    path("accounts/register/", views.register, name="register"),
]
