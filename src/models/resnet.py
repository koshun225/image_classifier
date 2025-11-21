"""
ResNetモデルの定義

torchvision.models.resnetを使用してResNetベースのモデルを構築します。
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResNetClassifier(nn.Module):
    """
    ResNetベースの画像分類モデル
    
    Args:
        model_name: モデル名 ("ResNet18", "ResNet50", "ResNet101", "ResNet152")
        num_classes: クラス数
        pretrained: 事前学習済みモデルを使用するか
        freeze_backbone: バックボーンを凍結するか
    """
    
    def __init__(
        self,
        model_name: str = "ResNet18",
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # ResNetモデルの取得
        if model_name == "ResNet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet18(weights=weights)
        elif model_name == "ResNet34":
            if pretrained:
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet34(weights=weights)
        elif model_name == "ResNet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet50(weights=weights)
        elif model_name == "ResNet101":
            if pretrained:
                weights = models.ResNet101_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet101(weights=weights)
        elif model_name == "ResNet152":
            if pretrained:
                weights = models.ResNet152_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet152(weights=weights)
        else:
            raise ValueError(
                f"サポートされていないモデル名: {model_name}. "
                f"サポートされているモデル: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152"
            )
        
        # 最終層の置き換え
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # バックボーンの凍結
        if freeze_backbone:
            logger.info("バックボーンを凍結します")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 最終層のみ学習可能にする
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        
        logger.info(
            f"ResNetモデル作成: {model_name}, "
            f"num_classes={num_classes}, "
            f"pretrained={pretrained}, "
            f"freeze_backbone={freeze_backbone}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル [batch_size, channels, height, width]
        
        Returns:
            出力テンソル [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def get_num_parameters(self) -> int:
        """
        パラメータ数を取得
        
        Returns:
            パラメータ数
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """
        学習可能なパラメータ数を取得
        
        Returns:
            学習可能なパラメータ数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"num_classes={self.num_classes}, "
            f"pretrained={self.pretrained}, "
            f"freeze_backbone={self.freeze_backbone}, "
            f"total_params={self.get_num_parameters():,}, "
            f"trainable_params={self.get_num_trainable_parameters():,})"
        )


if __name__ == "__main__":
    # テスト
    model = ResNetClassifier(
        model_name="ResNet18",
        num_classes=10,
        pretrained=True,
        freeze_backbone=False
    )
    
    print(model)
    
    # ダミー入力でテスト
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
