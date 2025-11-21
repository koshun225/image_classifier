#!/usr/bin/env python
"""
「未分割のみ」機能のテストスクリプト

データ分割時に「未分割のみ」と「全データ再分割」が正しく動作するかテスト
"""

import os
import sys
from pathlib import Path

# Django環境のセットアップ
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src' / 'web'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from data_management.crud import (
    get_theme,
    get_split_statistics,
    assign_splits_to_new_data,
    assign_all_splits,
    reset_splits
)


def test_unsplit_only_mode(theme_id: int):
    """未分割のみモードのテスト"""
    print("\n" + "="*60)
    print("テスト1: 未分割のみモード")
    print("="*60)
    
    # 初期状態を確認
    print("\n初期状態:")
    stats = get_split_statistics(theme_id)
    print(f"  Train: {stats['train']}枚")
    print(f"  Valid: {stats['valid']}枚")
    print(f"  Test: {stats['test']}枚")
    print(f"  未分割: {stats['unsplit']}枚")
    
    if stats['unsplit'] == 0:
        print("\n⚠️  未分割データがないため、分割をリセットします...")
        reset_splits(theme_id)
        stats = get_split_statistics(theme_id)
        print(f"  リセット後 - 未分割: {stats['unsplit']}枚")
    
    # 未分割データのみを分割
    print("\n未分割データのみを分割中...")
    new_stats = assign_splits_to_new_data(
        theme_id=theme_id,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print("\n分割後:")
    print(f"  Train: {new_stats['train']}枚")
    print(f"  Valid: {new_stats['valid']}枚")
    print(f"  Test: {new_stats['test']}枚")
    print(f"  未分割: {new_stats['unsplit']}枚")
    
    # 検証
    if new_stats['unsplit'] == 0:
        print("\n✅ テスト1成功: 未分割データがすべて分割されました")
    else:
        print(f"\n❌ テスト1失敗: まだ未分割データが{new_stats['unsplit']}枚残っています")
    
    return new_stats


def test_resplit_all_mode(theme_id: int):
    """全データ再分割モードのテスト"""
    print("\n" + "="*60)
    print("テスト2: 全データ再分割モード")
    print("="*60)
    
    # 現在の状態を確認
    print("\n再分割前の状態:")
    stats_before = get_split_statistics(theme_id)
    print(f"  Train: {stats_before['train']}枚")
    print(f"  Valid: {stats_before['valid']}枚")
    print(f"  Test: {stats_before['test']}枚")
    print(f"  未分割: {stats_before['unsplit']}枚")
    
    # 全データを再分割（異なる比率で）
    print("\n全データを再分割中（比率を変更: 60:20:20）...")
    new_stats = assign_all_splits(
        theme_id=theme_id,
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        random_seed=99
    )
    
    print("\n再分割後:")
    print(f"  Train: {new_stats['train']}枚")
    print(f"  Valid: {new_stats['valid']}枚")
    print(f"  Test: {new_stats['test']}枚")
    print(f"  未分割: {new_stats['unsplit']}枚")
    
    # 検証
    total_before = stats_before['train'] + stats_before['valid'] + stats_before['test'] + stats_before['unsplit']
    total_after = new_stats['train'] + new_stats['valid'] + new_stats['test'] + new_stats['unsplit']
    
    # 比率をチェック
    expected_train = int(total_after * 0.6)
    expected_valid = int(total_after * 0.2)
    
    train_diff = abs(new_stats['train'] - expected_train)
    valid_diff = abs(new_stats['valid'] - expected_valid)
    
    print(f"\n比率検証:")
    print(f"  Train: {new_stats['train']}枚 (期待値: 約{expected_train}枚, 差分: {train_diff}枚)")
    print(f"  Valid: {new_stats['valid']}枚 (期待値: 約{expected_valid}枚, 差分: {valid_diff}枚)")
    
    if total_before == total_after and new_stats['unsplit'] == 0:
        print("\n✅ テスト2成功: 全データが新しい比率で再分割されました")
        return True
    else:
        print(f"\n❌ テスト2失敗: データ数が一致しません（前: {total_before}, 後: {total_after}）")
        return False


def test_idempotency(theme_id: int):
    """べき等性のテスト"""
    print("\n" + "="*60)
    print("テスト3: べき等性テスト（未分割のみモード）")
    print("="*60)
    
    # 1回目の分割
    print("\n1回目の分割...")
    stats1 = assign_splits_to_new_data(
        theme_id=theme_id,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print(f"  Train: {stats1['train']}枚")
    print(f"  Valid: {stats1['valid']}枚")
    print(f"  Test: {stats1['test']}枚")
    print(f"  未分割: {stats1['unsplit']}枚")
    
    # 2回目の分割（未分割がない場合は何も変わらないはず）
    print("\n2回目の分割（未分割データがない状態）...")
    stats2 = assign_splits_to_new_data(
        theme_id=theme_id,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print(f"  Train: {stats2['train']}枚")
    print(f"  Valid: {stats2['valid']}枚")
    print(f"  Test: {stats2['test']}枚")
    print(f"  未分割: {stats2['unsplit']}枚")
    
    # 検証
    if stats1 == stats2:
        print("\n✅ テスト3成功: べき等性が保証されています（2回実行しても同じ結果）")
        return True
    else:
        print("\n❌ テスト3失敗: べき等性が保証されていません")
        return False


def main(theme_id: int):
    """メイン処理"""
    print("============================================================")
    print("「未分割のみ」機能のテスト")
    print("============================================================")
    
    theme = get_theme(theme_id)
    if not theme:
        print(f"❌ エラー: テーマID {theme_id} が見つかりません。")
        return
    
    print(f"\n✓ テーマ: {theme.name} (ID: {theme.id})")
    
    # テスト実行
    try:
        # テスト1: 未分割のみモード
        test_unsplit_only_mode(theme_id)
        
        # テスト2: 全データ再分割モード
        test_resplit_all_mode(theme_id)
        
        # テスト3: べき等性テスト
        test_idempotency(theme_id)
        
        print("\n" + "="*60)
        print("すべてのテストが完了しました")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="「未分割のみ」機能のテストスクリプト")
    parser.add_argument("--theme-id", type=int, default=7, help="テストするテーマのID")
    args = parser.parse_args()
    
    main(args.theme_id)

