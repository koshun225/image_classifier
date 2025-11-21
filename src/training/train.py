"""
学習スクリプト

Lightning Trainerの初期化、Callbacksの設定、MLflowLoggerの設定、学習の実行を行います。
"""

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import mlflow
import mlflow.pytorch

from src.data.datamodule import ClassificationDataModule
from src.training.lightning_module import ClassificationLightningModule
from src.training.callbacks import get_default_callbacks
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_model_metadata,
    save_and_log_params,
    get_git_commit_id,
)
from src.utils.dvc_utils import get_data_version_from_config
from src.utils.params_schema import materialize_params

logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> Dict[str, Any]:
    """
    params.yamlを読み込む
    
    Args:
        params_file: params.yamlファイルのパス
    
    Returns:
        パラメータの辞書
    """
    with open(params_file, "r") as f:
        schema = yaml.safe_load(f)
    
    params = materialize_params(schema or {})
    logger.info(f"パラメータを読み込みました: {params_file}")
    return params


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    config.yamlを読み込む
    
    Args:
        config_file: config.yamlファイルのパス
    
    Returns:
        設定の辞書
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"設定を読み込みました: {config_file}")
    return config


def train(
    params_file: str = "params.yaml",
    config_file: str = "config.yaml",
    augments_config: str = "auguments.yaml",
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    enable_mlflow: bool = True,
    run_name: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    学習を実行
    
    Args:
        params_file: params.yamlファイルのパス
        config_file: config.yamlファイルのパス
        augments_config: auguments.yamlファイルのパス
        checkpoint_dir: チェックポイントの保存ディレクトリ
        log_dir: ログの保存ディレクトリ
        enable_mlflow: MLflowを使用するか
        run_name: MLflow run名
        mlflow_run_id: 既存のMLflow run ID（指定した場合はそのrunを使用）
        **kwargs: その他のパラメータ
    
    Returns:
        学習結果の辞書
    """
    # パラメータと設定の読み込み
    params = load_params(params_file)
    config = load_config(config_file)
    
    # パラメータの展開
    model_config = params.get("model", {})
    training_config = params.get("training", {})
    data_config = params.get("data", {})
    
    # theme_idを取得
    theme_id = data_config.get("theme_id")
    if theme_id is None:
        raise ValueError("theme_idがparams.yamlのdataセクションに設定されていません")
    
    # Django環境のセットアップとテーマ名の取得
    import sys
    import os
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / 'src' / 'web'))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    import django
    django.setup()
    
    from data_management.crud import get_theme
    theme = get_theme(theme_id=theme_id)
    if theme is None:
        raise ValueError(f"テーマID {theme_id} が見つかりません")
    
    theme_name = theme.name
    logger.info(f"テーマ '{theme_name}' (ID: {theme_id}) のデータを使用して学習を開始します")
    
    # MLflowのセットアップ（実験名をテーマ名に設定）
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "experiments/mlruns")
    experiment_name = theme_name  # テーマ名を実験名として使用
    
    # run_nameが指定されていない場合、params.yamlから読み込む
    if run_name is None:
        run_name = training_config.get("run_name")
    
    if enable_mlflow:
        setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
        logger.info(f"MLflow実験名: '{experiment_name}'")
        if run_name:
            logger.info(f"MLflow run名: '{run_name}'")
    
    # DataModuleの作成（Djangoベース）
    batch_size = training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 4)
    
    datamodule = ClassificationDataModule(
        theme_id=theme_id,
        augments_config=augments_config,
        batch_size=batch_size,
        num_workers=num_workers,
        use_preprocessing=kwargs.get("use_preprocessing", False)
    )
    
    # DataModuleのセットアップ（クラス数の取得のため）
    datamodule.setup("fit")
    num_classes = datamodule.get_num_classes()
    class_names = datamodule.get_class_names()
    
    logger.info(f"クラス数: {num_classes}")
    logger.info(f"クラス名: {class_names}")
    
    # モデル設定の更新
    model_config["num_classes"] = num_classes
    
    # チェックポイントパスの取得
    checkpoint_path = training_config.get("checkpoint_path")
    
    # LightningModuleの作成
    model = ClassificationLightningModule(
        model_name=model_config.get("name", "ResNet18"),
        num_classes=num_classes,
        learning_rate=training_config.get("learning_rate", 0.001),
        optimizer=training_config.get("optimizer", "Adam"),
        weight_decay=training_config.get("weight_decay", 0.0),
        momentum=training_config.get("momentum", 0.9),
        scheduler=training_config.get("scheduler"),
        scheduler_params=training_config.get("scheduler_params"),
        pretrained=model_config.get("pretrained", True),
        freeze_backbone=model_config.get("freeze_backbone", False)
    )
    
    # チェックポイントから重みを読み込む
    if checkpoint_path:
        try:
            # チェックポイントパスの処理
            # MLflow artifact pathの場合: "mlflow-artifacts:/..." または "file:///path/to/artifacts/model"
            # ファイルシステムパスの場合: "/path/to/artifacts/model"
            if checkpoint_path.startswith('mlflow-artifacts:'):
                # MLflow Tracking Serverを使用している場合
                # mlflow-artifacts:/experiments/mlruns/{experiment_id}/{run_id}/artifacts/model の形式
                # 実際のファイルパスに変換
                artifact_path = checkpoint_path.replace('mlflow-artifacts:', '')
                # 先頭のスラッシュを削除して、project_rootからの相対パスとして扱う
                artifact_path = artifact_path.lstrip('/')
                # experiments/mlruns/{experiment_id}/{run_id}/artifacts/model の形式
                checkpoint_file = project_root / artifact_path / 'model.pth'
            elif checkpoint_path.startswith('file://'):
                checkpoint_file = Path(checkpoint_path[7:]) / 'model.pth'
            else:
                # ファイルシステムパスの場合（既にmodelディレクトリを指している）
                checkpoint_file = Path(checkpoint_path) / 'model.pth'
            
            # チェックポイントファイルが存在するか確認
            if checkpoint_file.exists():
                logger.info(f"チェックポイントから重みを読み込みます: {checkpoint_file}")
                # PyTorchモデルを読み込む
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                
                # チェックポイントがLightningModuleの形式か、通常のPyTorchモデルの形式かを判定
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # LightningModuleのチェックポイント形式
                    state_dict = checkpoint['state_dict']
                    # 'model.'プレフィックスを削除（LightningModuleのstate_dictには'model.'が付く場合がある）
                    model_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('model.'):
                            model_state_dict[k[6:]] = v
                        else:
                            model_state_dict[k] = v
                    model.model.load_state_dict(model_state_dict, strict=False)
                else:
                    # 通常のPyTorchモデルの形式
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        # ネストされた形式
                        model.model.load_state_dict(checkpoint['model'], strict=False)
                    else:
                        # 直接モデルのstate_dict
                        model.model.load_state_dict(checkpoint, strict=False)
                
                logger.info("チェックポイントから重みを読み込みました")
            else:
                logger.warning(f"チェックポイントファイルが見つかりません: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"チェックポイントの読み込みに失敗しました: {e}")
            logger.warning("新しいモデルとして学習を開始します")
    
    # Callbacksの作成
    callbacks = get_default_callbacks(
        checkpoint_dir=checkpoint_dir,
        monitor=kwargs.get("monitor", "val_loss"),
        patience=training_config.get("patience", 10)
    )
    
    # Loggerの作成
    loggers = []
    mlflow_logger = None
    
    if enable_mlflow:
        if mlflow_run_id:
            # 既存のrunを使用
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_id=mlflow_run_id,
                log_model=False  # 手動でログする
            )
            logger.info(f"既存のMLflow run ID={mlflow_run_id}を使用します")
        else:
            # 新しいrunを作成
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_name=run_name,
                log_model=False  # 手動でログする
            )
        loggers.append(mlflow_logger)
    
    # Trainerの作成
    num_epochs = training_config.get("num_epochs", 100)
    accelerator = kwargs.get("accelerator", "auto")
    devices = kwargs.get("devices", "auto")
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=loggers if loggers else None,
        callbacks=callbacks,
        deterministic=kwargs.get("deterministic", True),
        log_every_n_steps=kwargs.get("log_every_n_steps", 10),
        val_check_interval=kwargs.get("val_check_interval", 1.0),
        gradient_clip_val=training_config.get("gradient_clip_val"),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        precision=kwargs.get("precision", "32-true"),
    )
    
    logger.info(f"Trainer作成: max_epochs={num_epochs}, accelerator={accelerator}, devices={devices}")
    
    # 学習の実行
    logger.info("学習を開始します...")
    trainer.fit(model, datamodule)
    
    # テストの実行
    logger.info("テストを実行します...")
    test_results = trainer.test(model, datamodule)
    
    # MLflowへの追加ログ記録（MLFlowLoggerのrunコンテキストを使用）
    if enable_mlflow and mlflow_logger:
        logger.info("追加情報とモデルをMLflowに保存します...")
        
        # MLFlowLoggerが既にrunをアクティブにしているので、
        # mlflow_run_idが指定されている場合は新たにstart_runしない
        if mlflow_run_id:
            # 既存のrunを使用している場合は、直接mlflow APIを使う
            # テーマ情報のログ
            mlflow.log_param("theme_id", theme_id)
            mlflow.log_param("theme_name", theme_name)
            
            # メタデータのログ
            data_version = get_data_version_from_config(config_file)
            log_model_metadata(
                data_version=data_version,
                config_file=config_file,
                data_dvc_file="data.dvc"
            )
            
            # パラメータの保存とログ
            save_and_log_params(params, save_path="params_used.yaml")
            
            # auguments.yamlのログ
            mlflow.log_artifact(augments_config, artifact_path="config")
            
            # 最良のチェックポイントをロード
            if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                best_model_path = trainer.checkpoint_callback.best_model_path
                logger.info(f"最良のチェックポイント: {best_model_path}")
                model = ClassificationLightningModule.load_from_checkpoint(best_model_path)
            
            # PyTorchモデルを取得
            pytorch_model = model.get_model()
            
            # モデルをMLflowにログ（レジストリには登録しない）
            mlflow.pytorch.log_model(
                pytorch_model=pytorch_model,
                artifact_path="model",
                registered_model_name=None  # レジストリには登録しない
            )
            
            # クラス名を保存
            class_names_path = Path("class_names.txt")
            with open(class_names_path, "w") as f:
                for class_name in class_names:
                    f.write(f"{class_name}\n")
            mlflow.log_artifact(str(class_names_path), artifact_path="model")
            class_names_path.unlink()  # 削除
            
            logger.info("モデルをMLflowに保存しました（レジストリには登録していません）")
        else:
            # 新しいrunの場合は、明示的にrunコンテキストを使用
            with mlflow.start_run(run_id=mlflow_logger.run_id):
                # テーマ情報のログ
                mlflow.log_param("theme_id", theme_id)
                mlflow.log_param("theme_name", theme_name)
                
                # メタデータのログ
                data_version = get_data_version_from_config(config_file)
                log_model_metadata(
                    data_version=data_version,
                    config_file=config_file,
                    data_dvc_file="data.dvc"
                )
                
                # パラメータの保存とログ
                save_and_log_params(params, save_path="params_used.yaml")
                
                # auguments.yamlのログ
                mlflow.log_artifact(augments_config, artifact_path="config")
                
                # 最良のチェックポイントをロード
                if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                    best_model_path = trainer.checkpoint_callback.best_model_path
                    logger.info(f"最良のチェックポイント: {best_model_path}")
                    model = ClassificationLightningModule.load_from_checkpoint(best_model_path)
                
                # PyTorchモデルを取得
                pytorch_model = model.get_model()
                
                # モデルをMLflowにログ
                mlflow.pytorch.log_model(
                    pytorch_model=pytorch_model,
                    artifact_path="model",
                    registered_model_name=None  # 後でregister_model.pyで登録
                )
                
                # クラス名を保存
                class_names_path = Path("class_names.txt")
                with open(class_names_path, "w") as f:
                    for class_name in class_names:
                        f.write(f"{class_name}\n")
                mlflow.log_artifact(str(class_names_path), artifact_path="model")
                class_names_path.unlink()  # 削除
                
                logger.info("モデルをMLflowに保存しました")
    
    # 学習結果
    results = {
        "test_results": test_results,
        "best_model_path": trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None,
        "mlflow_run_id": mlflow_logger.run_id if mlflow_logger else None,
        "trainer_callback_metrics": dict(trainer.callback_metrics),  # train/validのメトリクス
    }
    
    logger.info("学習が完了しました")
    return results


if __name__ == "__main__":
    # テスト用
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s"
    )
    
    results = train(
        params_file="params.yaml",
        config_file="config.yaml",
        augments_config="auguments.yaml",
        enable_mlflow=True,
        run_name="test_run"
    )
    
    print("学習結果:")
    print(f"  テスト結果: {results['test_results']}")
    print(f"  最良モデルパス: {results['best_model_path']}")
    print(f"  MLflow run ID: {results['mlflow_run_id']}")
