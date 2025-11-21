"""
Lightning Callbacks

カスタムCallbacksの定義と学習制御を行います。
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    TQDMProgressBar
)
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_callbacks(
    checkpoint_dir: str = "checkpoints",
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 3,
    enable_early_stopping: bool = True,
    enable_model_checkpoint: bool = True,
    enable_lr_monitor: bool = True,
    enable_progress_bar: bool = True,
    progress_bar_type: str = "tqdm",  # "tqdm" or "rich"
    **kwargs
) -> List[pl.Callback]:
    """
    Callbacksを作成
    
    Args:
        checkpoint_dir: チェックポイントの保存ディレクトリ
        monitor: モニターするメトリクス
        mode: モニターモード（"min" or "max"）
        patience: Early Stoppingの待機エポック数
        save_top_k: 保存する上位チェックポイント数
        enable_early_stopping: Early Stoppingを有効にするか
        enable_model_checkpoint: ModelCheckpointを有効にするか
        enable_lr_monitor: LearningRateMonitorを有効にするか
        enable_progress_bar: プログレスバーを有効にするか
        progress_bar_type: プログレスバーのタイプ（"tqdm" or "rich"）
        **kwargs: その他のパラメータ
    
    Returns:
        Callbacksのリスト
    """
    callbacks = []
    
    # ModelCheckpoint
    if enable_model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
        logger.info(
            f"ModelCheckpoint: dirpath={checkpoint_dir}, "
            f"monitor={monitor}, mode={mode}, save_top_k={save_top_k}"
        )
    
    # Early Stopping
    if enable_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True,
            min_delta=kwargs.get("min_delta", 0.0),
            check_on_train_epoch_end=False
        )
        callbacks.append(early_stop_callback)
        logger.info(
            f"EarlyStopping: monitor={monitor}, mode={mode}, patience={patience}"
        )
    
    # LearningRateMonitor
    if enable_lr_monitor:
        lr_monitor = LearningRateMonitor(
            logging_interval="epoch"
        )
        callbacks.append(lr_monitor)
        logger.info("LearningRateMonitor enabled")
    
    # ProgressBar
    if enable_progress_bar:
        if progress_bar_type == "rich":
            try:
                progress_bar = RichProgressBar()
                callbacks.append(progress_bar)
                logger.info("RichProgressBar enabled")
            except ImportError:
                logger.warning("Rich not installed, falling back to TQDM")
                progress_bar = TQDMProgressBar()
                callbacks.append(progress_bar)
        else:
            progress_bar = TQDMProgressBar()
            callbacks.append(progress_bar)
            logger.info("TQDMProgressBar enabled")
    
    return callbacks


class MetricsLoggerCallback(pl.Callback):
    """
    メトリクスをログに記録するカスタムCallback
    
    エポック終了時にメトリクスをログに出力します。
    """
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        学習エポック終了時
        """
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # ログに記録
        log_msg = f"Epoch {epoch}: "
        log_msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if "train" in k])
        logger.info(log_msg)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        検証エポック終了時
        """
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # ログに記録
        log_msg = f"Epoch {epoch} [Val]: "
        log_msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if "val" in k])
        logger.info(log_msg)


class GradientNormCallback(pl.Callback):
    """
    勾配ノルムをモニタリングするカスタムCallback
    
    学習中の勾配ノルムをログに記録し、勾配爆発/消失を検知します。
    """
    
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        逆伝播後
        """
        # 勾配ノルムの計算
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # ログに記録
        pl_module.log("grad_norm", total_norm, on_step=True, on_epoch=False)


def get_default_callbacks(
    checkpoint_dir: str = "checkpoints",
    monitor: str = "val_loss",
    patience: int = 10,
    **kwargs
) -> List[pl.Callback]:
    """
    デフォルトのCallbacksを取得
    
    Args:
        checkpoint_dir: チェックポイントの保存ディレクトリ
        monitor: モニターするメトリクス
        patience: Early Stoppingの待機エポック数
        **kwargs: その他のパラメータ
    
    Returns:
        Callbacksのリスト
    """
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        patience=patience,
        **kwargs
    )
    
    # カスタムCallbacksを追加
    callbacks.append(MetricsLoggerCallback())
    # callbacks.append(GradientNormCallback())  # 必要に応じて有効化
    
    return callbacks


if __name__ == "__main__":
    # テスト
    callbacks = get_default_callbacks(
        checkpoint_dir="checkpoints",
        monitor="val_loss",
        patience=10
    )
    
    print("作成されたCallbacks:")
    for i, callback in enumerate(callbacks):
        print(f"  {i+1}. {callback.__class__.__name__}")
