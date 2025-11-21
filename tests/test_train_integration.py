#!/usr/bin/env python
"""
学習統合テストスクリプト

Djangoベースのデータレイヤーと学習レイヤーの統合が正しく動作するか確認
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Django環境のセットアップ
sys.path.insert(0, str(project_root / 'src' / 'web'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from data_management.crud import get_theme, get_split_statistics


def test_datamodule_creation(theme_id: int):
    """DataModuleが正常に作成できるかテスト"""
    print("\n" + "="*60)
    print("テスト1: DataModule作成テスト")
    print("="*60)
    
    try:
        from src.data.datamodule import ClassificationDataModule
        
        datamodule = ClassificationDataModule(
            theme_id=theme_id,
            augments_config="auguments.yaml",
            batch_size=4,
            num_workers=0,
            use_preprocessing=False
        )
        
        print("✅ DataModuleの作成成功")
        return datamodule
    except Exception as e:
        print(f"❌ DataModuleの作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_datamodule_setup(datamodule, theme_id: int):
    """DataModuleのセットアップが正常に動作するかテスト"""
    print("\n" + "="*60)
    print("テスト2: DataModuleセットアップテスト")
    print("="*60)
    
    if datamodule is None:
        print("❌ DataModuleが作成されていません")
        return False
    
    try:
        datamodule.setup("fit")
        
        num_classes = datamodule.get_num_classes()
        class_names = datamodule.get_class_names()
        
        print(f"✅ セットアップ成功")
        print(f"  クラス数: {num_classes}")
        print(f"  クラス名: {class_names}")
        print(f"  Train dataset: {len(datamodule.train_dataset)}枚")
        print(f"  Valid dataset: {len(datamodule.val_dataset)}枚")
        
        return True
    except Exception as e:
        print(f"❌ セットアップ失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(datamodule):
    """DataLoaderが正常に動作するかテスト"""
    print("\n" + "="*60)
    print("テスト3: DataLoaderテスト")
    print("="*60)
    
    if datamodule is None:
        print("❌ DataModuleが作成されていません")
        return False
    
    try:
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        # 最初のバッチを取得
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        print(f"✅ DataLoader動作成功")
        print(f"  Train batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label values: {labels}")
        
        return True
    except Exception as e:
        print(f"❌ DataLoader失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation(num_classes: int):
    """モデルが正常に作成できるかテスト"""
    print("\n" + "="*60)
    print("テスト4: モデル作成テスト")
    print("="*60)
    
    try:
        from src.models.model_factory import create_model
        
        model = create_model(
            model_name="ResNet18",
            num_classes=num_classes,
            pretrained=False
        )
        
        print(f"✅ モデル作成成功")
        print(f"  モデル: ResNet18")
        print(f"  クラス数: {num_classes}")
        
        return model
    except Exception as e:
        print(f"❌ モデル作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_lightning_module(model, num_classes: int):
    """LightningModuleが正常に作成できるかテスト"""
    print("\n" + "="*60)
    print("テスト5: LightningModule作成テスト")
    print("="*60)
    
    if model is None:
        print("❌ モデルが作成されていません")
        return False
    
    try:
        from src.training.lightning_module import ClassificationLightningModule
        
        lightning_module = ClassificationLightningModule(
            model=model,
            num_classes=num_classes,
            learning_rate=0.001,
            optimizer="Adam"
        )
        
        print(f"✅ LightningModule作成成功")
        return True
    except Exception as e:
        print(f"❌ LightningModule作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(theme_id: int):
    """メイン処理"""
    print("============================================================")
    print("学習統合テスト")
    print("============================================================")
    
    # テーマの確認
    theme = get_theme(theme_id)
    if not theme:
        print(f"❌ エラー: テーマID {theme_id} が見つかりません。")
        return
    
    print(f"\n✓ テーマ: {theme.name} (ID: {theme.id})")
    
    # データ統計の確認
    stats = get_split_statistics(theme_id)
    print(f"\nデータ統計:")
    print(f"  Train: {stats['train']}枚")
    print(f"  Valid: {stats['valid']}枚")
    print(f"  Test: {stats['test']}枚")
    print(f"  未分割: {stats['unsplit']}枚")
    
    if stats['train'] == 0 or stats['valid'] == 0:
        print("\n❌ エラー: 学習に必要なデータが不足しています。")
        print("   データ分割を実行してください。")
        return
    
    # テスト実行
    datamodule = test_datamodule_creation(theme_id)
    
    if datamodule:
        success = test_datamodule_setup(datamodule, theme_id)
        
        if success:
            test_dataloader(datamodule)
            
            num_classes = datamodule.get_num_classes()
            model = test_model_creation(num_classes)
            
            if model:
                test_lightning_module(model, num_classes)
    
    print("\n" + "="*60)
    print("テスト完了")
    print("="*60)
    print("\n次のステップ:")
    print("1. 実際の学習を実行:")
    print(f"   python scripts/train.py --theme-id {theme_id} --epochs 2 --batch-size 4")
    print("\n2. MLflowなしで軽量実行:")
    print(f"   python scripts/train.py --theme-id {theme_id} --epochs 1 --no-mlflow")
    print("\n✓ すべての準備が整いました！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="学習統合テスト")
    parser.add_argument("--theme-id", type=int, default=7, help="テストするテーマのID")
    args = parser.parse_args()
    
    main(args.theme_id)

