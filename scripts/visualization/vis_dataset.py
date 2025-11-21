"""
Dataset と DataModule の動作確認デモ

Dataset、DataLoader、DataModule の動作を確認します。
Djangoデータベースからテーマのデータを取得して使用します。
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Django環境のセットアップ
sys.path.insert(0, str(project_root / 'src' / 'web'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from src.data.dataset import ClassificationDataset
from src.data.datamodule import ClassificationDataModule
from src.data.augmentation import get_transforms
from src.data.preprocessing import create_preprocessing_pipeline
from data_management.crud import get_theme, get_traindata_by_theme


def demo_dataset_basic(theme_id: int, split: str = "train"):
    """Datasetの基本動作確認"""
    print("\n" + "=" * 60)
    print("Dataset 基本動作確認")
    print("=" * 60)
    
    # テーマの確認
    theme = get_theme(theme_id=theme_id)
    if theme is None:
        print(f"\nエラー: テーマID {theme_id} が見つかりません")
        return None
    
    print(f"\nテーマ情報:")
    print(f"  ID: {theme.id}")
    print(f"  名前: {theme.name}")
    
    # 分割データの確認
    train_data_list = get_traindata_by_theme(theme_id=theme_id, split=split)
    if not train_data_list:
        print(f"\nエラー: テーマID {theme_id} に{split}データがありません")
        print("先にDjango管理画面でデータ分割を実行してください")
        return None
    
    print(f"  {split}データ数: {len(train_data_list)}枚")
    
    # auguments.yamlから設定を読み込み
    augments_file = Path("auguments.yaml")
    preprocessing_pipeline = None
    
    if augments_file.exists():
        print(f"\n✓ {augments_file} から設定を読み込み中...")
        with open(augments_file, "r") as f:
            config = yaml.safe_load(f)
            preprocessing_config = config.get("preprocessing", {})
            image_config = config.get("image", {})
            
            # 前処理パイプラインを作成
            if preprocessing_config:
                preprocessing_pipeline = create_preprocessing_pipeline(
                    preprocessing_config,
                    image_config=image_config
                )
                if preprocessing_pipeline:
                    print("  前処理パイプラインを作成しました")
                    print(f"  {preprocessing_pipeline}")
                else:
                    print("  有効な前処理が設定されていません")
    
    # 学習用変換を取得
    transform = get_transforms("auguments.yaml", split=split)
    
    # Datasetを作成（Django theme_idベース）
    dataset = ClassificationDataset(
        theme_id=theme_id,
        split=split,
        transform=transform,
        preprocessing=preprocessing_pipeline
    )
    
    print(f"\nDataset情報:")
    print(dataset)
    print(f"サンプル数: {len(dataset)}")
    print(f"クラス数: {len(dataset.class_to_idx)}")
    print(f"クラス名: {list(dataset.class_to_idx.keys())}")
    
    # クラス分布
    distribution = dataset.get_class_distribution()
    print(f"\nクラス分布:")
    for class_idx, count in sorted(distribution.items()):
        class_name = dataset.idx_to_class[class_idx]
        print(f"  クラス {class_name}: {count}サンプル")
    
    # 最初のサンプルを取得
    image, label = dataset[0]
    print(f"\n最初のサンプル:")
    print(f"  画像テンソル形状: {image.shape}")
    print(f"  ラベル: {label}")
    print(f"  データ型: {image.dtype}")
    print(f"  値の範囲: [{image.min():.3f}, {image.max():.3f}]")
    
    return dataset


def demo_dataset_visualization(dataset, num_samples=16):
    """Datasetのサンプルを可視化"""
    print("\n" + "=" * 60)
    print("Dataset サンプル可視化")
    print("=" * 60)
    
    if dataset is None:
        print("Datasetが作成されていません")
        return
    
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        try:
            image, label = dataset[i]
            
            # テンソルをnumpy配列に変換 (C, H, W) -> (H, W, C)
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
                # 正規化を元に戻す（おおよそ）
                # 値が[-1, 1]や[0, 1]の範囲の場合、[0, 1]にクリップ
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = image
            
            axes[row, col].imshow(image_np)
            class_name = dataset.idx_to_class[label]
            axes[row, col].set_title(f"Class {class_name}")
            axes[row, col].axis("off")
        except Exception as e:
            print(f"エラー (Sample {i}): {e}")
            axes[row, col].axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_dataset_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存しました: {output_path}")
    plt.show()


def demo_dataloader(dataset):
    """DataLoaderの動作確認"""
    print("\n" + "=" * 60)
    print("DataLoader 動作確認")
    print("=" * 60)
    
    if dataset is None:
        print("Datasetが作成されていません")
        return
    
    from torch.utils.data import DataLoader
    
    # DataLoaderを作成
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # デモ用に0
        drop_last=False
    )
    
    print(f"DataLoader情報:")
    print(f"  バッチサイズ: {loader.batch_size}")
    print(f"  総バッチ数: {len(loader)}")
    print(f"  シャッフル: {loader._iterator is None or hasattr(loader._iterator, '_index_sampler')}")
    
    # 最初のバッチを取得
    images, labels = next(iter(loader))
    
    print(f"\n最初のバッチ:")
    print(f"  画像テンソル形状: {images.shape}")  # (B, C, H, W)
    print(f"  ラベルテンソル形状: {labels.shape}")  # (B,)
    print(f"  ラベル: {labels.tolist()}")
    
    # バッチを可視化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i in range(4):
        image = images[i].permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        label = labels[i].item()
        
        axes[i].imshow(image)
        class_name = dataset.idx_to_class[label]
        axes[i].set_title(f"Class {class_name}")
        axes[i].axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_dataloader_batch.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存しました: {output_path}")
    plt.show()


def demo_datamodule(theme_id: int):
    """DataModuleの動作確認"""
    print("\n" + "=" * 60)
    print("DataModule 動作確認")
    print("=" * 60)
    
    # テーマの確認
    theme = get_theme(theme_id=theme_id)
    if theme is None:
        print(f"\nエラー: テーマID {theme_id} が見つかりません")
        return
    
    print(f"\nテーマ情報:")
    print(f"  ID: {theme.id}")
    print(f"  名前: {theme.name}")
    
    # DataModuleを作成（Django theme_idベース）
    dm = ClassificationDataModule(
        theme_id=theme_id,
        augments_config="auguments.yaml",
        batch_size=4,
        num_workers=0,  # デモ用に0
        use_preprocessing=True  # 前処理を有効化
    )
    
    print(f"\nDataModule情報:")
    print(dm)
    
    # セットアップ
    print("\nセットアップ中...")
    dm.setup("fit")
    
    print(f"\n各データセットのサイズ:")
    print(f"  学習: {len(dm.train_dataset)}サンプル")
    print(f"  検証: {len(dm.val_dataset)}サンプル")
    
    dm.setup("test")
    print(f"  テスト: {len(dm.test_dataset)}サンプル")
    
    # クラス情報
    num_classes = dm.get_num_classes()
    class_names = dm.get_class_names()
    print(f"\nクラス情報:")
    print(f"  クラス数: {num_classes}")
    print(f"  クラス名: {class_names}")
    
    # DataLoaderを取得
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print(f"\nDataLoader情報:")
    print(f"  学習: {len(train_loader)}バッチ")
    print(f"  検証: {len(val_loader)}バッチ")
    
    # サンプルバッチを取得
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    
    print(f"\n学習バッチ:")
    print(f"  形状: {train_images.shape}")
    print(f"  ラベル: {train_labels.tolist()}")
    
    print(f"\n検証バッチ:")
    print(f"  形状: {val_images.shape}")
    print(f"  ラベル: {val_labels.tolist()}")
    
    # 学習と検証のサンプルを可視化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 学習サンプル
    for i in range(min(4, len(train_labels))):
        image = train_images[i].permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        label = train_labels[i].item()
        
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Train - Class {class_names[label]}")
        axes[0, i].axis("off")
    
    # 検証サンプル
    for i in range(min(4, len(val_labels))):
        image = val_images[i].permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        label = val_labels[i].item()
        
        axes[1, i].imshow(image)
        axes[1, i].set_title(f"Val - Class {class_names[label]}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_datamodule_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 保存しました: {output_path}")
    plt.show()


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description="Dataset & DataModule の動作確認デモ（Django統合版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme-id",
        type=int,
        required=True,
        help="テーマID"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="可視化するサンプル数"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset & DataModule デモスクリプト（Django統合版）")
    print("=" * 60)
    
    # 前提条件の確認
    if not Path("auguments.yaml").exists():
        print("\nエラー: auguments.yaml が見つかりません")
        return 1
    
    print("\n✓ auguments.yaml から設定を読み込みます")
    
    try:
        # 各デモを実行
        dataset = demo_dataset_basic(args.theme_id, split="train")
        if dataset is None:
            return 1
        
        demo_dataset_visualization(dataset, num_samples=args.num_samples)
        demo_dataloader(dataset)
        demo_datamodule(args.theme_id)
        
        print("\n" + "=" * 60)
        print("完了")
        print("=" * 60)
        print("\n生成された画像:")
        print("  - workspace/demo_dataset_samples.png")
        print("  - workspace/demo_dataloader_batch.png")
        print("  - workspace/demo_datamodule_samples.png")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)


