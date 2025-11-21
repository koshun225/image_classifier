"""
データベースモデル定義

Theme, Label, TrainData, Model, ModelTrainData, TrainingJob を定義
"""
from django.db import models
from django.core.validators import FileExtensionValidator
from django.utils import timezone
import json


class Theme(models.Model):
    """テーマ（分類タスク）"""
    name = models.CharField(max_length=200, unique=True, verbose_name="テーマ名")
    description = models.TextField(blank=True, null=True, verbose_name="説明")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="作成日時")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新日時")

    class Meta:
        verbose_name = "テーマ"
        verbose_name_plural = "テーマ"
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def get_label_count(self):
        """ラベル数を取得"""
        return self.labels.count()

    def get_image_count(self):
        """画像数を取得"""
        return self.traindata.count()


class Label(models.Model):
    """ラベル（クラス）"""
    theme = models.ForeignKey(Theme, on_delete=models.CASCADE, related_name='labels', verbose_name="テーマ")
    label_name = models.CharField(max_length=100, verbose_name="ラベル名")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="作成日時")

    class Meta:
        verbose_name = "ラベル"
        verbose_name_plural = "ラベル"
        unique_together = [['theme', 'label_name']]
        ordering = ['id']

    def __str__(self):
        return f"{self.theme.name} - {self.label_name}"


class TrainData(models.Model):
    """学習データ（画像）"""
    SPLIT_CHOICES = [
        ('train', 'Train'),
        ('valid', 'Valid'),
        ('test', 'Test'),
    ]

    theme = models.ForeignKey(Theme, on_delete=models.CASCADE, related_name='traindata', verbose_name="テーマ")
    label = models.ForeignKey(Label, on_delete=models.CASCADE, null=True, blank=True, verbose_name="ラベル")
    image = models.ImageField(
        upload_to='images/%Y/%m/%d/',
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'bmp', 'gif'])],
        verbose_name="画像"
    )
    split = models.CharField(max_length=10, choices=SPLIT_CHOICES, blank=True, null=True, verbose_name="データ分割")
    labeled_by = models.CharField(max_length=100, blank=True, null=True, verbose_name="ラベル付けした人")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="作成日時")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新日時")

    class Meta:
        verbose_name = "学習データ"
        verbose_name_plural = "学習データ"
        ordering = ['-created_at']

    def __str__(self):
        label_name = self.label.label_name if self.label else "未ラベル"
        return f"{self.theme.name} - {label_name} - {self.image.name}"


class Model(models.Model):
    """学習済みモデル"""
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    theme = models.ForeignKey(Theme, on_delete=models.CASCADE, related_name='models', verbose_name="テーマ")
    mlflow_run_id = models.CharField(max_length=200, unique=True, verbose_name="MLflow Run ID")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='completed', verbose_name="ステータス")
    created_by = models.CharField(max_length=100, blank=True, null=True, verbose_name="作成者")
    training_params = models.TextField(blank=True, null=True, verbose_name="学習パラメータ（JSON）")
    best_score = models.FloatField(null=True, blank=True, verbose_name="最良スコア")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="作成日時")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新日時")

    class Meta:
        verbose_name = "モデル"
        verbose_name_plural = "モデル"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.theme.name} - {self.mlflow_run_id}"
    
    def get_training_params_dict(self):
        """training_paramsを辞書として取得"""
        if self.training_params:
            try:
                return json.loads(self.training_params)
            except json.JSONDecodeError:
                return {}
        return {}


class ModelTrainData(models.Model):
    """モデルと学習データの関連（中間テーブル）"""
    model = models.ForeignKey(Model, on_delete=models.CASCADE, related_name='train_data_relations', verbose_name="モデル")
    train_data = models.ForeignKey(TrainData, on_delete=models.CASCADE, related_name='model_relations', verbose_name="学習データ")
    created_at = models.DateTimeField(default=timezone.now, verbose_name="作成日時")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新日時")

    class Meta:
        verbose_name = "モデル-学習データ関連"
        verbose_name_plural = "モデル-学習データ関連"
        unique_together = [['model', 'train_data']]

    def __str__(self):
        return f"{self.model} - {self.train_data}"


class TrainingJob(models.Model):
    """学習ジョブ（バックグラウンド実行の管理）"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    theme = models.ForeignKey(Theme, on_delete=models.CASCADE, related_name='training_jobs', verbose_name="テーマ")
    model = models.ForeignKey(Model, on_delete=models.SET_NULL, null=True, blank=True, related_name='training_jobs', verbose_name="モデル")
    process_id = models.IntegerField(null=True, blank=True, verbose_name="プロセスID")
    mlflow_parent_run_id = models.CharField(max_length=200, blank=True, null=True, verbose_name="MLflow 親ランID")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="ステータス")
    log_file = models.CharField(max_length=500, blank=True, null=True, verbose_name="ログファイルパス")
    params_yaml = models.TextField(blank=True, null=True, verbose_name="params.yaml")
    optuna_params_yaml = models.TextField(blank=True, null=True, verbose_name="optuna設定")
    auguments_yaml = models.TextField(blank=True, null=True, verbose_name="auguments.yaml")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="開始日時")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="完了日時")
    error_message = models.TextField(blank=True, null=True, verbose_name="エラーメッセージ")
    created_by = models.CharField(max_length=100, blank=True, null=True, verbose_name="作成者")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="作成日時")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新日時")

    class Meta:
        verbose_name = "学習ジョブ"
        verbose_name_plural = "学習ジョブ"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.theme.name} - {self.status} - {self.created_at}"
