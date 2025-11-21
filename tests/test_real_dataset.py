"""
実データを使用したデータセットテスト

本番データベース（theme_id=7）を使用してデータセットの動作を確認
PyTorchなしで実行可能
"""

import pytest
from pathlib import Path
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src' / 'web'))

# Django環境をセットアップ
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()


def test_real_theme_exists():
    """実際のテーマ（MNIST Test）が存在するか"""
    from data_management.crud import get_theme
    
    theme = get_theme(theme_id=7)
    assert theme is not None, "テーマID 7 が見つかりません"
    assert theme.name == "MNIST Test", f"テーマ名が不正: {theme.name}"
    
    print(f"✓ テーマ確認成功: {theme.name} (ID: {theme.id})")


def test_real_labels_exist():
    """実際のラベル（0-9）が存在するか"""
    from data_management.crud import get_labels_by_theme, get_class_names_by_theme
    
    labels = get_labels_by_theme(theme_id=7)
    assert len(labels) == 10, f"ラベル数が不正: {len(labels)}"
    
    class_names = get_class_names_by_theme(theme_id=7)
    expected_names = [str(i) for i in range(10)]
    assert class_names == expected_names, f"クラス名が不正: {class_names}"
    
    print(f"✓ ラベル確認成功: {len(labels)}個")
    print(f"  クラス名: {class_names}")


def test_real_data_split():
    """実際のデータ分割が正しく行われているか"""
    from data_management.crud import get_split_statistics
    
    stats = get_split_statistics(theme_id=7)
    
    assert 'train' in stats, "train統計がありません"
    assert 'valid' in stats, "valid統計がありません"
    assert 'test' in stats, "test統計がありません"
    assert 'unsplit' in stats, "unsplit統計がありません"
    
    total = stats['train'] + stats['valid'] + stats['test'] + stats['unsplit']
    
    print(f"✓ データ分割統計:")
    print(f"  Train: {stats['train']}枚")
    print(f"  Valid: {stats['valid']}枚")
    print(f"  Test: {stats['test']}枚")
    print(f"  未分割: {stats['unsplit']}枚")
    print(f"  合計: {total}枚")
    
    assert total > 0, "データが1枚も登録されていません"
    assert stats['train'] > 0, "訓練データが登録されていません"
    assert stats['valid'] > 0, "検証データが登録されていません"
    assert stats['test'] > 0, "テストデータが登録されていません"


def test_real_traindata_retrieval():
    """実際の学習データが取得できるか"""
    from data_management.crud import get_traindata_by_theme
    
    # すべてのデータを取得
    all_data = get_traindata_by_theme(theme_id=7)
    assert len(all_data) > 0, "学習データが取得できません"
    
    # Train分割のデータを取得
    train_data = get_traindata_by_theme(theme_id=7, split='train')
    assert len(train_data) > 0, "Train分割のデータが取得できません"
    
    # Valid分割のデータを取得
    valid_data = get_traindata_by_theme(theme_id=7, split='valid')
    assert len(valid_data) > 0, "Valid分割のデータが取得できません"
    
    # Test分割のデータを取得
    test_data = get_traindata_by_theme(theme_id=7, split='test')
    assert len(test_data) > 0, "Test分割のデータが取得できません"
    
    print(f"✓ データ取得成功:")
    print(f"  全データ: {len(all_data)}枚")
    print(f"  Train: {len(train_data)}枚")
    print(f"  Valid: {len(valid_data)}枚")
    print(f"  Test: {len(test_data)}枚")
    
    # サンプルデータの確認
    sample = train_data[0]
    assert sample.image is not None, "画像が登録されていません"
    assert sample.label is not None, "ラベルが登録されていません"
    assert sample.split == 'train', f"分割が不正: {sample.split}"
    
    print(f"\n✓ サンプルデータ:")
    print(f"  ID: {sample.id}")
    print(f"  ラベル: {sample.label.label_name}")
    print(f"  分割: {sample.split}")
    print(f"  画像: {Path(sample.image.name).name}")


def test_real_split_consistency():
    """データ分割の一貫性を確認"""
    from data_management.crud import get_traindata_by_theme, get_split_statistics
    
    stats = get_split_statistics(theme_id=7)
    
    # 各分割のデータを取得
    train_data = get_traindata_by_theme(theme_id=7, split='train')
    valid_data = get_traindata_by_theme(theme_id=7, split='valid')
    test_data = get_traindata_by_theme(theme_id=7, split='test')
    
    # 統計と実データの数が一致するか確認
    assert len(train_data) == stats['train'], \
        f"Train統計が一致しません: {len(train_data)} != {stats['train']}"
    assert len(valid_data) == stats['valid'], \
        f"Valid統計が一致しません: {len(valid_data)} != {stats['valid']}"
    assert len(test_data) == stats['test'], \
        f"Test統計が一致しません: {len(test_data)} != {stats['test']}"
    
    # データIDの重複がないか確認
    train_ids = set(td.id for td in train_data)
    valid_ids = set(td.id for td in valid_data)
    test_ids = set(td.id for td in test_data)
    
    assert len(train_ids & valid_ids) == 0, "TrainとValidでデータIDが重複しています"
    assert len(train_ids & test_ids) == 0, "TrainとTestでデータIDが重複しています"
    assert len(valid_ids & test_ids) == 0, "ValidとTestでデータIDが重複しています"
    
    print(f"✓ データ分割の一貫性確認成功")
    print(f"  データIDの重複なし")
    print(f"  統計と実データの数が一致")


def test_real_class_distribution():
    """各分割におけるクラス分布を確認"""
    from data_management.crud import get_traindata_by_theme
    
    for split in ['train', 'valid', 'test']:
        data = get_traindata_by_theme(theme_id=7, split=split)
        
        # クラスごとの画像数をカウント
        class_counts = {}
        for td in data:
            label_name = td.label.label_name
            class_counts[label_name] = class_counts.get(label_name, 0) + 1
        
        print(f"\n✓ {split.capitalize()}のクラス分布:")
        for label_name in sorted(class_counts.keys(), key=lambda x: int(x)):
            count = class_counts[label_name]
            print(f"  クラス '{label_name}': {count}枚")
        
        # すべてのクラスが含まれているか確認（理想的には）
        # ただし、データ量が少ない場合は一部のクラスがない可能性もある
        assert len(class_counts) > 0, f"{split}分割にデータがありません"


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v", "-s"])

