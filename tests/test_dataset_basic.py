"""
データセット基本機能のテスト（Django部分のみ）
PyTorchなしで実行可能な基本テスト
"""

import pytest
from pathlib import Path
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_django_setup(setup_django_env):
    """Django環境がセットアップできるか"""
    import django
    assert django.VERSION is not None
    print(f"✓ Django version: {django.get_version()}")


def test_create_theme_and_labels(test_theme):
    """テーマとラベルが作成できるか"""
    theme, labels = test_theme
    
    assert theme is not None, "テーマが作成されていません"
    assert theme.name == "MNIST Test", "テーマ名が正しくありません"
    assert len(labels) == 10, f"ラベル数が不正です: {len(labels)}"
    
    print(f"✓ テーマ作成成功: {theme.name}")
    print(f"✓ ラベル作成成功: {len(labels)}個")
    
    for label in labels:
        print(f"  - ラベル: {label.label_name}")


def test_database_operations(test_theme):
    """データベース操作が正常に動作するか"""
    from data_management.crud import (
        get_theme,
        get_labels_by_theme,
        get_class_names_by_theme,
        get_split_statistics
    )
    
    theme, labels = test_theme
    
    # テーマの取得
    retrieved_theme = get_theme(theme_id=theme.id)
    assert retrieved_theme is not None, "テーマが取得できません"
    assert retrieved_theme.id == theme.id, "テーマIDが一致しません"
    
    # ラベルの取得
    retrieved_labels = get_labels_by_theme(theme_id=theme.id)
    assert len(retrieved_labels) == 10, "ラベル数が一致しません"
    
    # クラス名の取得
    class_names = get_class_names_by_theme(theme_id=theme.id)
    assert len(class_names) == 10, "クラス名の数が一致しません"
    assert class_names == [str(i) for i in range(10)], "クラス名が正しくありません"
    
    # 分割統計の取得
    stats = get_split_statistics(theme_id=theme.id)
    assert 'train' in stats, "train統計がありません"
    assert 'valid' in stats, "valid統計がありません"
    assert 'test' in stats, "test統計がありません"
    assert 'unsplit' in stats, "unsplit統計がありません"
    
    print(f"✓ データベース操作成功")
    print(f"  - テーマ: {retrieved_theme.name}")
    print(f"  - ラベル数: {len(retrieved_labels)}")
    print(f"  - クラス名: {class_names}")
    print(f"  - 分割統計: {stats}")


def test_data_split_without_data(test_theme):
    """データがない状態での分割テスト"""
    from src.data.split import split_dataset, get_split_info
    
    theme, labels = test_theme
    
    # データがない状態で分割を実行
    stats = split_dataset(
        theme_id=theme.id,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # すべて0であるべき
    assert stats['train'] == 0, "データがないのにtrainが0でない"
    assert stats['valid'] == 0, "データがないのにvalidが0でない"
    assert stats['test'] == 0, "データがないのにtestが0でない"
    assert stats['unsplit'] == 0, "データがないのにunsplitが0でない"
    
    print(f"✓ データなし分割テスト成功: {stats}")


def test_dataset_info_without_data(test_theme):
    """データがない状態でのデータセット情報取得"""
    from src.data.dataset import get_dataset_info
    
    theme, labels = test_theme
    
    # データセット情報を取得
    info = get_dataset_info(theme_id=theme.id)
    
    assert info['theme_id'] == theme.id, "テーマIDが一致しません"
    assert info['theme_name'] == theme.name, "テーマ名が一致しません"
    assert info['num_classes'] == 10, "クラス数が一致しません"
    assert len(info['class_names']) == 10, "クラス名の数が一致しません"
    assert info['total_samples'] == 0, "データがないのにサンプル数が0でない"
    
    print(f"✓ データセット情報取得成功:")
    print(f"  - テーマ: {info['theme_name']}")
    print(f"  - クラス数: {info['num_classes']}")
    print(f"  - 総サンプル数: {info['total_samples']}")


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v", "-s"])

