"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from data_management import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # 認証
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # テーマ関連
    path('', views.theme_list, name='theme_list'),
    path('theme/create/', views.theme_create, name='theme_create'),
    path('theme/<int:theme_id>/', views.theme_detail, name='theme_detail'),
    
    # REST API
    path('api/theme/<int:theme_id>/label/update/<int:traindata_id>/', views.api_update_label, name='api_update_label'),
    path('api/theme/<int:theme_id>/images/upload/', views.api_upload_images, name='api_upload_images'),
    path('api/theme/<int:theme_id>/images/<int:traindata_id>/', views.api_delete_image, name='api_delete_image'),
    path('api/theme/<int:theme_id>/split/', views.api_split_data, name='api_split_data'),
    path('api/theme/<int:theme_id>/statistics/', views.api_statistics, name='api_statistics'),
    
    # モデル開発関連
    path('theme/<int:theme_id>/model/development/', views.model_development, name='model_development'),
    path('theme/<int:theme_id>/models/', views.model_list, name='model_list'),
    path('theme/<int:theme_id>/training/<int:job_id>/', views.model_training_status, name='model_training_status'),
    path('api/theme/<int:theme_id>/model/<int:model_id>/params/', views.api_get_model_params, name='api_get_model_params'),
    path('api/theme/<int:theme_id>/params/save/', views.api_save_params, name='api_save_params'),
    path('api/theme/<int:theme_id>/preview/augmentation/', views.api_preview_augmentation, name='api_preview_augmentation'),
    path('api/theme/<int:theme_id>/training/start/', views.api_start_training, name='api_start_training'),
    path('api/theme/<int:theme_id>/training/status/<int:job_id>/', views.api_training_status, name='api_training_status'),
]

# メディアファイルの配信（開発環境のみ）
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
