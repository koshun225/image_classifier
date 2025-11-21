#!/usr/bin/env python
"""
修正後のマルチプロセステスト

DataLoaderのマルチプロセスでpickle可能なデータ構造をテスト
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


def test_dataset_pickle():
    """Datasetがpickle可能かテスト"""
    print("\n" + "="*60)
    print("テスト1: Datasetのpickle可能性テスト")
    print("="*60)
    
    try:
        import pickle
        from src.data.dataset import ClassificationDataset
        
        dataset = ClassificationDataset(
            theme_id=7,
            split='train',
            transform=None,
            preprocessing=None
        )
        
        print(f"✅ Dataset作成成功: {len(dataset)}サンプル")
        
        # pickleテスト
        pickled = pickle.dumps(dataset)
        unpickled_dataset = pickle.loads(pickled)
        
        print(f"✅ Pickle/Unpickle成功")
        print(f"  元のサンプル数: {len(dataset)}")
        print(f"  復元後のサンプル数: {len(unpickled_dataset)}")
        
        # 1つ目のサンプルを取得してテスト
        image, label = dataset[0]
        print(f"✅ データ取得成功: image shape={image.shape}, label={label}")
        
        return True
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_workers():
    """DataLoaderのマルチプロセステスト"""
    print("\n" + "="*60)
    print("テスト2: DataLoaderマルチプロセステスト")
    print("="*60)
    
    try:
        from torch.utils.data import DataLoader
        from src.data.dataset import ClassificationDataset
        
        dataset = ClassificationDataset(
            theme_id=7,
            split='train',
            transform=None,
            preprocessing=None
        )
        
        # num_workers=2でDataLoader作成
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2  # マルチプロセス
        )
        
        print(f"✅ DataLoader作成成功 (num_workers=2)")
        
        # 最初のバッチを取得
        batch = next(iter(dataloader))
        images, labels = batch
        
        print(f"✅ バッチ取得成功")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        
        return True
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン処理"""
    print("="*60)
    print("修正後のマルチプロセステスト")
    print("="*60)
    print("\n修正内容:")
    print("- Djangoモデルインスタンスをpickle可能な辞書に変換")
    print("- 画像パスとラベル名のみを保持")
    print("- DataLoaderのマルチプロセスに対応\n")
    
    success1 = test_dataset_pickle()
    
    if success1:
        success2 = test_dataloader_workers()
    else:
        success2 = False
    
    print("\n" + "="*60)
    print("テスト結果")
    print("="*60)
    print(f"1. Pickle可能性: {'✅ 成功' if success1 else '❌ 失敗'}")
    print(f"2. マルチプロセス: {'✅ 成功' if success2 else '❌ 失敗'}")
    
    if success1 and success2:
        print("\n✅ すべてのテストが成功しました！")
        print("\n次のステップ:")
        print("  python scripts/train.py --theme-id 7 --epochs 2 --batch-size 4")
    else:
        print("\n❌ テストに失敗しました。")


if __name__ == "__main__":
    main()

