"""
統合テスト: データセット読み込みからモデル出力まで（Djangoデータベースベース）

データベースベースのデータセットの読み込み、DataModule、LightningModule、
モデルの順伝播までの一連の流れをテストします。
"""

import pytest
import torch
import yaml
from pathlib import Path

from src.data.dataset import ClassificationDataset, load_dataset, get_dataset_info
from src.data.datamodule import ClassificationDataModule
from src.data.split import split_dataset
from src.data.augmentation import get_transforms
from src.training.lightning_module import ClassificationLightningModule
from src.models.resnet import ResNetClassifier
from src.models.model_factory import create_model


class TestDatasetLoading:
    """データセット読み込みのテスト"""
    
    def test_dataset_from_database(self, mnist_test_data, params):
        """データベースからDatasetを作成できるか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        # Datasetの作成
        dataset = ClassificationDataset(
            theme_id=theme.id,
            split='train',
            transform=None,
            preprocessing=None
        )
        
        # 基本的な検証
        assert len(dataset) > 0, "データセットが空です"
        assert len(dataset.class_to_idx) > 0, "クラスが見つかりません"
        assert len(dataset.class_to_idx) == 10, "MNISTのクラス数は10であるべき"
        
        # 1つのサンプルを取得
        image, label = dataset[0]
        
        # 画像のチェック
        assert isinstance(image, torch.Tensor), "画像がTensorではありません"
        assert image.ndim == 3, f"画像の次元が不正です: {image.ndim}"
        assert image.shape[0] == 3, f"チャンネル数が不正です: {image.shape[0]}"
        
        # ラベルのチェック
        assert isinstance(label, int), "ラベルがintではありません"
        assert 0 <= label < len(dataset.class_to_idx), "ラベルが範囲外です"
    
    def test_dataset_with_transforms(self, mnist_test_data, params, project_root):
        """変換を適用したDatasetが動作するか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        # 変換の取得
        augments_config = project_root / "auguments.yaml"
        transform = get_transforms(str(augments_config), split="train")
        
        # Datasetの作成
        dataset = ClassificationDataset(
            theme_id=theme.id,
            split='train',
            transform=transform,
            preprocessing=None
        )
        
        # サンプルを取得
        image, label = dataset[0]
        
        # 画像のチェック
        assert isinstance(image, torch.Tensor), "画像がTensorではありません"
        assert image.ndim == 3, "画像の次元が不正です"
    
    def test_load_dataset_helper(self, mnist_test_data, params):
        """load_dataset便利関数が動作するか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        # load_dataset便利関数を使用
        dataset = load_dataset(theme_id=theme.id, split='train')
        
        assert len(dataset) > 0, "データセットが空です"
        assert len(dataset.class_to_idx) == 10, "クラス数が不正です"
    
    def test_get_dataset_info_helper(self, mnist_test_data, params):
        """get_dataset_info便利関数が動作するか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        # get_dataset_info便利関数を使用
        info = get_dataset_info(theme_id=theme.id)
        
        assert info['theme_id'] == theme.id, "テーマIDが一致しません"
        assert info['num_classes'] == 10, "クラス数が不正です"
        assert info['total_samples'] > 0, "総サンプル数が0です"


class TestDataModule:
    """DataModuleのテスト"""
    
    def test_datamodule_setup(self, mnist_test_data, params, project_root):
        """DataModuleのセットアップが正常に動作するか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        augments_config = project_root / "auguments.yaml"
        
        # DataModuleの作成
        datamodule = ClassificationDataModule(
            theme_id=theme.id,
            augments_config=str(augments_config),
            batch_size=4,
            num_workers=0,  # テスト時は0
            use_preprocessing=False
        )
        
        # セットアップ
        datamodule.setup("fit")
        
        # 検証
        assert datamodule.train_dataset is not None, "訓練データセットが作成されていません"
        assert datamodule.val_dataset is not None, "検証データセットが作成されていません"
        assert len(datamodule.train_dataset) > 0, "訓練データセットが空です"
        assert len(datamodule.val_dataset) > 0, "検証データセットが空です"
        
        # クラス数の取得
        num_classes = datamodule.get_num_classes()
        assert num_classes > 0, "クラス数が0です"
        assert num_classes == 10, "MNISTのクラス数は10であるべき"
        
        # クラス名の取得
        class_names = datamodule.get_class_names()
        assert len(class_names) == num_classes, "クラス名の数が一致しません"
    
    def test_datamodule_dataloaders(self, mnist_test_data, params, project_root):
        """DataLoaderが正常に動作するか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        augments_config = project_root / "auguments.yaml"
        
        # DataModuleの作成
        datamodule = ClassificationDataModule(
            theme_id=theme.id,
            augments_config=str(augments_config),
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        # セットアップ
        datamodule.setup("fit")
        
        # DataLoaderの取得
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        # 訓練DataLoaderのチェック
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        assert images.shape[0] <= 4, "バッチサイズが不正です"
        assert images.ndim == 4, "画像の次元が不正です"
        assert labels.shape[0] == images.shape[0], "ラベル数が画像数と一致しません"
        
        # 検証DataLoaderのチェック
        val_batch = next(iter(val_loader))
        images, labels = val_batch
        
        assert images.shape[0] <= 4, "バッチサイズが不正です"
        assert images.ndim == 4, "画像の次元が不正です"
    
    def test_datamodule_test_split(self, mnist_test_data, params, project_root):
        """テスト用DataLoaderが正常に動作するか"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        augments_config = project_root / "auguments.yaml"
        
        # DataModuleの作成
        datamodule = ClassificationDataModule(
            theme_id=theme.id,
            augments_config=str(augments_config),
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        # セットアップ（test用）
        datamodule.setup("test")
        
        # テストDataLoaderの取得
        test_loader = datamodule.test_dataloader()
        
        # テストDataLoaderのチェック
        test_batch = next(iter(test_loader))
        images, labels = test_batch
        
        assert images.shape[0] <= 4, "バッチサイズが不正です"
        assert images.ndim == 4, "画像の次元が不正です"


class TestModelCreation:
    """モデル作成のテスト"""
    
    @pytest.mark.parametrize("model_name", [
        "ResNet18",
        "ResNet34",
        "ResNet50",
    ])
    def test_create_model(self, model_name):
        """モデルが正常に作成できるか"""
        model = create_model(
            model_name=model_name,
            num_classes=10,
            pretrained=False,  # テスト時は高速化のためFalse
            freeze_backbone=False
        )
        
        assert model is not None, "モデルが作成されていません"
        assert isinstance(model, torch.nn.Module), "モデルがnn.Moduleではありません"
        
        # パラメータ数の確認
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "パラメータが存在しません"
    
    def test_model_forward(self):
        """モデルの順伝播が正常に動作するか"""
        model = ResNetClassifier(
            model_name="ResNet18",
            num_classes=10,
            pretrained=False,
            freeze_backbone=False
        )
        
        # ダミー入力
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        # 順伝播
        output = model(x)
        
        # 出力のチェック
        assert output.shape == (batch_size, 10), f"出力の形状が不正です: {output.shape}"
        assert not torch.isnan(output).any(), "出力にNaNが含まれています"
        assert not torch.isinf(output).any(), "出力にInfが含まれています"


class TestLightningModule:
    """LightningModuleのテスト"""
    
    def test_lightning_module_creation(self):
        """LightningModuleが正常に作成できるか"""
        module = ClassificationLightningModule(
            model_name="ResNet18",
            num_classes=10,
            learning_rate=0.001,
            optimizer="Adam",
            pretrained=False
        )
        
        assert module is not None, "LightningModuleが作成されていません"
        assert hasattr(module, "model"), "modelが存在しません"
    
    def test_lightning_module_forward(self):
        """LightningModuleの順伝播が正常に動作するか"""
        module = ClassificationLightningModule(
            model_name="ResNet18",
            num_classes=10,
            learning_rate=0.001,
            optimizer="Adam",
            pretrained=False
        )
        
        # ダミー入力
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        # 順伝播
        output = module(x)
        
        # 出力のチェック
        assert output.shape == (batch_size, 10), f"出力の形状が不正です: {output.shape}"
        assert not torch.isnan(output).any(), "出力にNaNが含まれています"
    
    def test_training_step(self):
        """training_stepが正常に動作するか"""
        module = ClassificationLightningModule(
            model_name="ResNet18",
            num_classes=10,
            learning_rate=0.001,
            optimizer="Adam",
            pretrained=False
        )
        
        # ダミーバッチ
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 10, (batch_size,))
        batch = (images, labels)
        
        # training_step
        loss = module.training_step(batch, batch_idx=0)
        
        # 損失のチェック
        assert loss is not None, "損失が返されていません"
        assert isinstance(loss, torch.Tensor), "損失がTensorではありません"
        assert loss.ndim == 0, "損失がスカラーではありません"
        assert not torch.isnan(loss), "損失がNaNです"
        assert loss.item() >= 0, "損失が負の値です"


class TestEndToEndPipeline:
    """エンドツーエンドパイプラインのテスト"""
    
    def test_full_pipeline(self, mnist_test_data, params, project_root):
        """データ読み込みからモデル出力までの全体の流れ"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        augments_config = project_root / "auguments.yaml"
        
        # 1. DataModuleの作成とセットアップ
        datamodule = ClassificationDataModule(
            theme_id=theme.id,
            augments_config=str(augments_config),
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        datamodule.setup("fit")
        
        num_classes = datamodule.get_num_classes()
        
        # 2. LightningModuleの作成
        module = ClassificationLightningModule(
            model_name="ResNet18",
            num_classes=num_classes,
            learning_rate=0.001,
            optimizer="Adam",
            pretrained=False
        )
        
        # 3. DataLoaderからバッチを取得
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        images, labels_batch = batch
        
        # 4. 順伝播
        output = module(images)
        
        # 5. 検証
        assert output.shape[0] == images.shape[0], "バッチサイズが一致しません"
        assert output.shape[1] == num_classes, "クラス数が一致しません"
        assert not torch.isnan(output).any(), "出力にNaNが含まれています"
        
        # 6. training_step
        loss = module.training_step(batch, batch_idx=0)
        assert loss is not None, "損失が計算されていません"
        assert not torch.isnan(loss), "損失がNaNです"
        
        print(f"✓ 統合テスト成功: num_classes={num_classes}, batch_size={images.shape[0]}, loss={loss.item():.4f}")
    
    def test_full_pipeline_with_preprocessing(self, mnist_test_data, params, project_root):
        """前処理を含む全体の流れ"""
        theme, labels = mnist_test_data
        
        # データを分割
        split_dataset(
            theme_id=theme.id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        augments_config = project_root / "auguments.yaml"
        
        # DataModuleの作成（前処理を有効化）
        datamodule = ClassificationDataModule(
            theme_id=theme.id,
            augments_config=str(augments_config),
            batch_size=4,
            num_workers=0,
            use_preprocessing=True  # 前処理を有効化
        )
        datamodule.setup("fit")
        
        num_classes = datamodule.get_num_classes()
        
        # LightningModuleの作成
        module = ClassificationLightningModule(
            model_name="ResNet18",
            num_classes=num_classes,
            learning_rate=0.001,
            optimizer="Adam",
            pretrained=False
        )
        
        # DataLoaderからバッチを取得
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        images, labels_batch = batch
        
        # 順伝播
        output = module(images)
        
        # 検証
        assert output.shape[0] == images.shape[0], "バッチサイズが一致しません"
        assert output.shape[1] == num_classes, "クラス数が一致しません"
        assert not torch.isnan(output).any(), "出力にNaNが含まれています"
        
        print(f"✓ 前処理付き統合テスト成功: num_classes={num_classes}, batch_size={images.shape[0]}")


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v", "-s"])
