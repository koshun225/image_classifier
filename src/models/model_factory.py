"""
モデルファクトリ

設定ファイルからモデルを生成します。
新しいモデルアーキテクチャを追加する場合は、このファイルに登録します。
"""

import torch.nn as nn
from typing import Dict, Any
import logging

from src.models.resnet import ResNetClassifier

logger = logging.getLogger(__name__)


# サポートされているモデルの登録
MODEL_REGISTRY = {
    "ResNet18": ResNetClassifier,
    "ResNet34": ResNetClassifier,
    "ResNet50": ResNetClassifier,
    "ResNet101": ResNetClassifier,
    "ResNet152": ResNetClassifier,
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    モデルを作成
    
    Args:
        model_name: モデル名（例: "ResNet18", "ResNet50"）
        num_classes: クラス数
        pretrained: 事前学習済みモデルを使用するか
        freeze_backbone: バックボーンを凍結するか
        **kwargs: その他のモデル固有のパラメータ
    
    Returns:
        PyTorchモデル
    
    Raises:
        ValueError: サポートされていないモデル名の場合
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"サポートされていないモデル名: {model_name}. "
            f"サポートされているモデル: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    
    # モデルの作成
    model = model_class(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **kwargs
    )
    
    logger.info(f"モデルを作成しました: {model_name}, num_classes={num_classes}")
    
    return model


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    設定辞書からモデルを作成
    
    Args:
        config: モデル設定の辞書
            例:
            {
                "model": {
                    "name": "ResNet18",
                    "num_classes": 10,
                    "pretrained": true,
                    "freeze_backbone": false
                }
            }
    
    Returns:
        PyTorchモデル
    """
    model_config = config.get("model", {})
    
    model_name = model_config.get("name", "ResNet18")
    num_classes = model_config.get("num_classes", 10)
    pretrained = model_config.get("pretrained", True)
    freeze_backbone = model_config.get("freeze_backbone", False)
    
    # その他のパラメータを抽出
    other_params = {
        k: v for k, v in model_config.items()
        if k not in ["name", "num_classes", "pretrained", "freeze_backbone"]
    }
    
    return create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **other_params
    )


def list_available_models():
    """
    利用可能なモデルのリストを取得
    
    Returns:
        モデル名のリスト
    """
    return list(MODEL_REGISTRY.keys())


def register_model(model_name: str, model_class):
    """
    新しいモデルを登録
    
    Args:
        model_name: モデル名
        model_class: モデルクラス
    """
    if model_name in MODEL_REGISTRY:
        logger.warning(f"モデル {model_name} は既に登録されています。上書きします。")
    
    MODEL_REGISTRY[model_name] = model_class
    logger.info(f"モデル {model_name} を登録しました")


if __name__ == "__main__":
    # テスト
    print("利用可能なモデル:")
    for model_name in list_available_models():
        print(f"  - {model_name}")
    
    # 設定からモデルを作成
    config = {
        "model": {
            "name": "ResNet18",
            "num_classes": 10,
            "pretrained": True,
            "freeze_backbone": False
        }
    }
    
    model = create_model_from_config(config)
    print(f"\n作成されたモデル:\n{model}")
