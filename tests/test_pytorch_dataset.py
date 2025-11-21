"""
PyTorchを使用した実データセットテスト

実際の画像読み込みとDataModuleの動作を確認
PyTorchがインストールされている必要があります
"""

import pytest
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# PyTorchのインポート
try:
    import torch
    import torchvision
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    pytest.skip("PyTorchがインストールされていません", allow_module_level=True)

from src.data.dataset import ClassificationDataset, load_dataset, get_dataset_info
from src.data.datamodule import ClassificationDataModule


class TestPyTorchDataset:
    """PyTorchを使用したデータセットテスト"""
    
    def test_dataset_creation(self):
        """データセットが正常に作成できるか"""
        dataset = ClassificationDataset(
            theme_id=7,
            split='train',
            transform=None,
            preprocessing=None
        )
        
        assert len(dataset) > 0, "データセットが空です"
        assert len(dataset.class_to_idx) == 10, "クラス数が10ではありません"
        
        print(f"✓ データセット作成成功")
        print(f"  サンプル数: {len(dataset)}枚")
        print(f"  クラス数: {len(dataset.class_to_idx)}")
        print(f"  クラス名: {list(dataset.class_to_idx.keys())}")
    
    def test_dataset_getitem(self):
        """データセットから画像が正常に取得できるか"""
        dataset = ClassificationDataset(
            theme_id=7,
            split='train',
            transform=None,
            preprocessing=None
        )
        
        # 最初のサンプルを取得
        image, label = dataset[0]
        
        # 画像のチェック
        assert isinstance(image, torch.Tensor), "画像がTensorではありません"
        assert image.ndim == 3, f"画像の次元が不正です: {image.ndim}"
        assert image.shape[0] == 3, f"チャンネル数が不正です: {image.shape[0]}"
        assert image.dtype == torch.float32, f"データ型が不正です: {image.dtype}"
        assert 0.0 <= image.min() <= 1.0, "画像の値が範囲外です（min）"
        assert 0.0 <= image.max() <= 1.0, "画像の値が範囲外です（max）"
        
        # ラベルのチェック
        assert isinstance(label, int), "ラベルがintではありません"
        assert 0 <= label < 10, f"ラベルが範囲外です: {label}"
        
        print(f"✓ データ取得成功")
        print(f"  画像shape: {image.shape}")
        print(f"  画像dtype: {image.dtype}")
        print(f"  画像値範囲: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  ラベル: {label}")
    
    def test_dataset_with_transform(self):
        """データオーグメンテーションが正常に動作するか"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = ClassificationDataset(
            theme_id=7,
            split='train',
            transform=transform,
            preprocessing=None
        )
        
        image, label = dataset[0]
        
        assert image.shape == (3, 224, 224), f"変換後のshapeが不正です: {image.shape}"
        
        print(f"✓ データオーグメンテーション成功")
        print(f"  変換後shape: {image.shape}")
    
    def test_load_dataset_helper(self):
        """load_dataset便利関数が動作するか"""
        dataset = load_dataset(theme_id=7, split='train')
        
        assert len(dataset) > 0, "データセットが空です"
        
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor), "画像がTensorではありません"
        
        print(f"✓ load_dataset便利関数成功")
        print(f"  サンプル数: {len(dataset)}枚")
    
    def test_get_dataset_info(self):
        """get_dataset_info関数が動作するか"""
        info = get_dataset_info(theme_id=7)
        
        assert info['theme_id'] == 7, "テーマIDが一致しません"
        assert info['num_classes'] == 10, "クラス数が不正です"
        assert info['total_samples'] > 0, "総サンプル数が0です"
        
        print(f"✓ データセット情報取得成功")
        print(f"  テーマ: {info['theme_name']}")
        print(f"  クラス数: {info['num_classes']}")
        print(f"  総サンプル数: {info['total_samples']}")
        print(f"  分割統計: {info['samples_per_split']}")


class TestPyTorchDataModule:
    """PyTorchを使用したDataModuleテスト"""
    
    def test_datamodule_creation(self):
        """DataModuleが正常に作成できるか"""
        datamodule = ClassificationDataModule(
            theme_id=7,
            augments_config="auguments.yaml",
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        assert datamodule is not None, "DataModuleが作成できません"
        
        print(f"✓ DataModule作成成功")
    
    def test_datamodule_setup(self):
        """DataModuleのセットアップが正常に動作するか"""
        datamodule = ClassificationDataModule(
            theme_id=7,
            augments_config="auguments.yaml",
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        # fitステージのセットアップ
        datamodule.setup('fit')
        
        assert datamodule.train_dataset is not None, "訓練データセットが作成されていません"
        assert datamodule.val_dataset is not None, "検証データセットが作成されていません"
        assert len(datamodule.train_dataset) > 0, "訓練データセットが空です"
        assert len(datamodule.val_dataset) > 0, "検証データセットが空です"
        
        num_classes = datamodule.get_num_classes()
        class_names = datamodule.get_class_names()
        
        assert num_classes == 10, "クラス数が10ではありません"
        assert len(class_names) == 10, "クラス名の数が10ではありません"
        
        print(f"✓ DataModuleセットアップ成功")
        print(f"  Train: {len(datamodule.train_dataset)}枚")
        print(f"  Valid: {len(datamodule.val_dataset)}枚")
        print(f"  クラス数: {num_classes}")
    
    def test_datamodule_dataloaders(self):
        """DataLoaderが正常に動作するか"""
        datamodule = ClassificationDataModule(
            theme_id=7,
            augments_config="auguments.yaml",
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        datamodule.setup('fit')
        
        # DataLoaderを取得
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        # 訓練DataLoaderのチェック
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        assert images.shape[0] <= 4, "バッチサイズが不正です"
        assert images.ndim == 4, f"画像の次元が不正です: {images.ndim}"
        assert images.shape[1] == 3, f"チャンネル数が不正です: {images.shape[1]}"
        assert labels.shape[0] == images.shape[0], "ラベル数が画像数と一致しません"
        
        print(f"✓ DataLoader動作成功")
        print(f"  Train batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # 検証DataLoaderのチェック
        val_batch = next(iter(val_loader))
        images, labels = val_batch
        
        assert images.shape[0] <= 4, "バッチサイズが不正です"
        assert images.ndim == 4, "画像の次元が不正です"
        
        print(f"  Valid batch shape: {images.shape}")
    
    def test_datamodule_test_split(self):
        """テスト用DataLoaderが正常に動作するか"""
        datamodule = ClassificationDataModule(
            theme_id=7,
            augments_config="auguments.yaml",
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        # testステージのセットアップ
        datamodule.setup('test')
        
        assert datamodule.test_dataset is not None, "テストデータセットが作成されていません"
        assert len(datamodule.test_dataset) > 0, "テストデータセットが空です"
        
        # DataLoaderを取得
        test_loader = datamodule.test_dataloader()
        
        # テストDataLoaderのチェック
        test_batch = next(iter(test_loader))
        images, labels = test_batch
        
        assert images.shape[0] <= 4, "バッチサイズが不正です"
        assert images.ndim == 4, "画像の次元が不正です"
        
        print(f"✓ Test DataLoader動作成功")
        print(f"  Test dataset: {len(datamodule.test_dataset)}枚")
        print(f"  Test batch shape: {images.shape}")


class TestEndToEndPipeline:
    """エンドツーエンドパイプラインテスト"""
    
    def test_full_pipeline(self):
        """データ読み込みから処理までの全体フロー"""
        # DataModuleを作成
        datamodule = ClassificationDataModule(
            theme_id=7,
            augments_config="auguments.yaml",
            batch_size=8,
            num_workers=0,
            use_preprocessing=False
        )
        
        # セットアップ
        datamodule.setup('fit')
        
        # DataLoaderを取得
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        # 複数バッチを処理
        train_batches = []
        for i, (images, labels) in enumerate(train_loader):
            train_batches.append((images.shape, labels.shape))
            if i >= 2:  # 最初の3バッチ
                break
        
        val_batches = []
        for i, (images, labels) in enumerate(val_loader):
            val_batches.append((images.shape, labels.shape))
            if i >= 1:  # 最初の2バッチ
                break
        
        print(f"✓ エンドツーエンドパイプライン成功")
        print(f"  Train batches: {len(train_batches)}個処理")
        for i, (img_shape, lbl_shape) in enumerate(train_batches):
            print(f"    Batch {i}: images={img_shape}, labels={lbl_shape}")
        
        print(f"  Valid batches: {len(val_batches)}個処理")
        for i, (img_shape, lbl_shape) in enumerate(val_batches):
            print(f"    Batch {i}: images={img_shape}, labels={lbl_shape}")
        
        assert len(train_batches) > 0, "訓練バッチが処理されていません"
        assert len(val_batches) > 0, "検証バッチが処理されていません"


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v", "-s"])

