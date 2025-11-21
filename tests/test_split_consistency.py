"""
データ分割の一貫性テスト（Djangoデータベースベース）

データベースに登録されたデータの分割一貫性をテストします。
- 初回分割と追加データ分割の一貫性
- 分割比率の正確性
- 未分割データの正しい処理
"""

import pytest
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.split import split_dataset, get_split_info, reset_splits
from src.data.dataset import get_dataset_info


TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42


class TestDataSplitConsistency:
    """データ分割の一貫性テスト"""
    
    def test_initial_split(self, mnist_test_data):
        """初回のデータ分割が正しく実行されるか"""
        theme, labels = mnist_test_data
        
        # 初回分割を実行
        stats = split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # 検証
        assert stats['train'] > 0, "訓練データが分割されていません"
        assert stats['valid'] > 0, "検証データが分割されていません"
        assert stats['test'] > 0, "テストデータが分割されていません"
        assert stats['unsplit'] == 0, "未分割データが残っています"
        
        # 分割比率の検証（許容誤差10%）
        total = stats['train'] + stats['valid'] + stats['test']
        train_actual_ratio = stats['train'] / total
        valid_actual_ratio = stats['valid'] / total
        test_actual_ratio = stats['test'] / total
        
        assert abs(train_actual_ratio - TRAIN_RATIO) < 0.1, \
            f"訓練データ比率が不正: 期待={TRAIN_RATIO}, 実際={train_actual_ratio}"
        assert abs(valid_actual_ratio - VALID_RATIO) < 0.1, \
            f"検証データ比率が不正: 期待={VALID_RATIO}, 実際={valid_actual_ratio}"
        assert abs(test_actual_ratio - TEST_RATIO) < 0.1, \
            f"テストデータ比率が不正: 期待={TEST_RATIO}, 実際={test_actual_ratio}"
    
    def test_split_idempotency(self, mnist_test_data):
        """同じシードで再分割しても結果が変わらないか"""
        theme, labels = mnist_test_data
        
        # 初回分割
        stats1 = split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # 分割情報を取得
        info1 = get_split_info(theme_id=theme.id)
        
        # 2回目の分割（未分割データがないので変化なし）
        stats2 = split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # 分割情報を取得
        info2 = get_split_info(theme_id=theme.id)
        
        # 検証: 分割結果が変わっていないこと
        assert stats1 == stats2, "分割結果が変わっています"
        assert info1 == info2, "分割情報が変わっています"
    
    def test_add_new_data_consistency(self, mnist_test_data, create_dummy_mnist_images):
        """新しいデータを追加しても既存の分割が保持されるか"""
        from data_management.crud import create_traindata, get_traindata_by_theme
        from django.core.files import File
        
        theme, labels = mnist_test_data
        
        # 初回分割を実行
        stats_before = split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # 各分割のデータIDを取得
        train_data_before = set(td.id for td in get_traindata_by_theme(theme.id, split='train'))
        valid_data_before = set(td.id for td in get_traindata_by_theme(theme.id, split='valid'))
        test_data_before = set(td.id for td in get_traindata_by_theme(theme.id, split='test'))
        
        # 新しいデータを追加
        with create_dummy_mnist_images(num_images_per_class=5) as (temp_dir, images):
            for class_id, image_paths in images.items():
                label = labels[class_id]
                for img_path in image_paths[:5]:  # クラスあたり5枚追加
                    with open(img_path, 'rb') as f:
                        django_file = File(f, name=Path(img_path).name)
                        create_traindata(
                            theme_id=theme.id,
                            image=django_file,
                            label_id=label.id,
                            labeled_by="test_user"
                        )
        
        # 再分割を実行
        stats_after = split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # 各分割のデータIDを取得
        train_data_after = set(td.id for td in get_traindata_by_theme(theme.id, split='train'))
        valid_data_after = set(td.id for td in get_traindata_by_theme(theme.id, split='valid'))
        test_data_after = set(td.id for td in get_traindata_by_theme(theme.id, split='test'))
        
        # 検証: 既存データの分割が保持されていること
        assert train_data_before.issubset(train_data_after), \
            "既存の訓練データが別の分割に移動しています"
        assert valid_data_before.issubset(valid_data_after), \
            "既存の検証データが別の分割に移動しています"
        assert test_data_before.issubset(test_data_after), \
            "既存のテストデータが別の分割に移動しています"
        
        # 検証: 新しいデータが追加されていること
        assert len(train_data_after) > len(train_data_before) or \
               len(valid_data_after) > len(valid_data_before) or \
               len(test_data_after) > len(test_data_before), \
            "新しいデータが分割に追加されていません"
        
        # 検証: 未分割データがないこと
        assert stats_after['unsplit'] == 0, "未分割データが残っています"
    
    def test_reset_splits(self, mnist_test_data):
        """分割のリセットが正しく動作するか"""
        theme, labels = mnist_test_data
        
        # 初回分割を実行
        stats_before = split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        assert stats_before['unsplit'] == 0, "初回分割後に未分割データがあります"
        
        # 分割をリセット
        stats_reset = reset_splits(theme_id=theme.id)
        
        # 検証: すべて未分割になっていること
        assert stats_reset['train'] == 0, "リセット後にtrainデータが残っています"
        assert stats_reset['valid'] == 0, "リセット後にvalidデータが残っています"
        assert stats_reset['test'] == 0, "リセット後にtestデータが残っています"
        assert stats_reset['unsplit'] > 0, "リセット後に未分割データがありません"
        
        # 総データ数が保持されていること
        total_before = stats_before['train'] + stats_before['valid'] + stats_before['test']
        total_reset = stats_reset['unsplit']
        assert total_before == total_reset, "リセット後のデータ数が一致しません"


class TestSplitStatistics:
    """分割統計のテスト"""
    
    def test_get_split_info(self, mnist_test_data):
        """分割統計が正しく取得できるか"""
        theme, labels = mnist_test_data
        
        # 分割を実行
        split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # 統計を取得
        stats = get_split_info(theme_id=theme.id)
        
        # 検証
        assert 'train' in stats, "訓練データ統計がありません"
        assert 'valid' in stats, "検証データ統計がありません"
        assert 'test' in stats, "テストデータ統計がありません"
        assert 'unsplit' in stats, "未分割データ統計がありません"
        
        total = stats['train'] + stats['valid'] + stats['test'] + stats['unsplit']
        assert total > 0, "総データ数が0です"
    
    def test_dataset_info(self, mnist_test_data):
        """データセット情報が正しく取得できるか"""
        theme, labels = mnist_test_data
        
        # 分割を実行
        split_dataset(
            theme_id=theme.id,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SPLIT_SEED
        )
        
        # データセット情報を取得
        info = get_dataset_info(theme_id=theme.id)
        
        # 検証
        assert info['theme_id'] == theme.id, "テーマIDが一致しません"
        assert info['theme_name'] == theme.name, "テーマ名が一致しません"
        assert info['num_classes'] == 10, "クラス数が一致しません"
        assert len(info['class_names']) == 10, "クラス名の数が一致しません"
        assert 'samples_per_split' in info, "分割統計がありません"
        assert info['total_samples'] > 0, "総サンプル数が0です"


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v", "-s"])
