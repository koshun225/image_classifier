"""
Django Admin設定
"""
from django.contrib import admin
from .models import Theme, Label, TrainData, Model, ModelTrainData


@admin.register(Theme)
class ThemeAdmin(admin.ModelAdmin):
    list_display = ['name', 'description', 'get_label_count', 'get_image_count', 'created_at']
    search_fields = ['name', 'description']
    list_filter = ['created_at']
    
    def get_label_count(self, obj):
        return obj.get_label_count()
    get_label_count.short_description = 'ラベル数'
    
    def get_image_count(self, obj):
        return obj.get_image_count()
    get_image_count.short_description = '画像数'


@admin.register(Label)
class LabelAdmin(admin.ModelAdmin):
    list_display = ['label_name', 'theme', 'created_at']
    list_filter = ['theme', 'created_at']
    search_fields = ['label_name', 'theme__name']


@admin.register(TrainData)
class TrainDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'theme', 'label', 'split', 'labeled_by', 'created_at']
    list_filter = ['theme', 'label', 'split', 'created_at']
    search_fields = ['theme__name', 'label__label_name', 'labeled_by']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    list_display = ['id', 'theme', 'mlflow_run_id', 'created_at']
    list_filter = ['theme', 'created_at']
    search_fields = ['theme__name', 'mlflow_run_id']


@admin.register(ModelTrainData)
class ModelTrainDataAdmin(admin.ModelAdmin):
    list_display = ['model', 'train_data']
    list_filter = ['model__theme']
