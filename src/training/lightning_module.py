"""
LightningModule定義

PyTorch LightningのLightningModuleを継承し、
モデルのforward、training_step、validation_step、test_stepを定義します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from typing import Dict, Any, Optional
import logging

from src.models.model_factory import create_model

logger = logging.getLogger(__name__)


class ClassificationLightningModule(pl.LightningModule):
    """
    画像分類用のLightningModule
    
    Args:
        model_name: モデル名（例: "ResNet18"）
        num_classes: クラス数
        learning_rate: 学習率
        optimizer: オプティマイザ名（"Adam", "SGD", "AdamW"）
        weight_decay: 重み減衰
        momentum: モーメンタム（SGDのみ）
        scheduler: スケジューラ名（"StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", None）
        scheduler_params: スケジューラのパラメータ
        pretrained: 事前学習済みモデルを使用するか
        freeze_backbone: バックボーンを凍結するか
    """
    
    def __init__(
        self,
        model_name: str = "ResNet18",
        num_classes: int = 10,
        learning_rate: float = 0.001,
        optimizer: str = "Adam",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # ハイパーパラメータの保存
        self.save_hyperparameters()
        
        # モデルの作成
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss()
        
        # メトリクス
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # Confusion Matrix（テスト時のみ）
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        logger.info(
            f"LightningModule作成: model={model_name}, "
            f"num_classes={num_classes}, "
            f"lr={learning_rate}, "
            f"optimizer={optimizer}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル [batch_size, channels, height, width]
        
        Returns:
            出力テンソル [batch_size, num_classes]
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        学習ステップ
        
        Args:
            batch: バッチデータ（images, labels）
            batch_idx: バッチインデックス
        
        Returns:
            損失値
        """
        images, labels = batch
        
        # 順伝播
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # メトリクスの計算
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_accuracy(preds, labels)
        
        # ログ
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        検証ステップ
        
        Args:
            batch: バッチデータ（images, labels）
            batch_idx: バッチインデックス
        
        Returns:
            損失値
        """
        images, labels = batch
        
        # 順伝播
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # メトリクスの計算
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        # ログ
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        テストステップ
        
        Args:
            batch: バッチデータ（images, labels）
            batch_idx: バッチインデックス
        
        Returns:
            損失値
        """
        images, labels = batch
        
        # 順伝播
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # メトリクスの計算
        preds = torch.argmax(outputs, dim=1)
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.test_confusion_matrix(preds, labels)
        
        # ログ
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """
        テストエポック終了時の処理
        
        Confusion Matrixをログに記録します。
        """
        cm = self.test_confusion_matrix.compute()
        logger.info(f"Test Confusion Matrix:\n{cm}")
        
        # MLflowにログ（MLFlowLoggerが使用可能な場合のみ）
        if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log_text'):
            try:
                self.logger.experiment.log_text(
                    run_id=self.logger.run_id,
                    text=str(cm.cpu().numpy()),
                    artifact_file="confusion_matrix.txt"
                )
            except Exception as e:
                logger.warning(f"Confusion Matrixのログに失敗しました: {e}")
    
    def configure_optimizers(self):
        """
        オプティマイザとスケジューラの設定
        
        Returns:
            オプティマイザまたは[オプティマイザ], [スケジューラ]
        """
        # オプティマイザの作成
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay
        
        if optimizer_name == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=lr,
                momentum=self.hparams.momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"サポートされていないオプティマイザ: {optimizer_name}")
        
        logger.info(f"オプティマイザ: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")
        
        # スケジューラの作成（オプション）
        scheduler_name = self.hparams.scheduler
        if scheduler_name is None:
            return optimizer
        
        scheduler_params = self.hparams.scheduler_params or {}
        
        if scheduler_name == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_params.get("step_size", 30),
                gamma=scheduler_params.get("gamma", 0.1)
            )
            return [optimizer], [scheduler]
        
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get("T_max", 100),
                eta_min=scheduler_params.get("eta_min", 0)
            )
            return [optimizer], [scheduler]
        
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params.get("mode", "min"),
                factor=scheduler_params.get("factor", 0.1),
                patience=scheduler_params.get("patience", 10),
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_params.get("monitor", "val_loss"),
                }
            }
        
        else:
            raise ValueError(f"サポートされていないスケジューラ: {scheduler_name}")
    
    def get_model(self) -> nn.Module:
        """
        内部のPyTorchモデルを取得
        
        Returns:
            PyTorchモデル
        """
        return self.model


if __name__ == "__main__":
    # テスト
    module = ClassificationLightningModule(
        model_name="ResNet18",
        num_classes=10,
        learning_rate=0.001,
        optimizer="Adam"
    )
    
    print(module)
    
    # ダミー入力でテスト
    x = torch.randn(4, 3, 224, 224)
    y = module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
