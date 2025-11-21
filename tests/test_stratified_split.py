#!/usr/bin/env python
"""
層化分割（Stratified Split）の動作確認スクリプト

既存のデータを使って、クラス割合が維持されているか検証します。
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Django環境のセットアップ
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src' / 'web'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from data_management.crud import (
    get_theme, 
    get_labels_by_theme,
    get_traindata_by_theme,
    get_split_statistics,
    assign_splits_to_new_data,
    reset_splits
)


def calculate_class_distribution(theme_id: int, split: str = None):
    """
    指定されたテーマの指定された分割におけるクラス分布を計算
    
    Args:
        theme_id: テーマID
        split: 分割名（None の場合は全データ）
    
    Returns:
        クラスごとのサンプル数と割合の辞書
    """
    traindata_list = get_traindata_by_theme(theme_id)
    
    if split:
        traindata_list = [td for td in traindata_list if td.split == split]
    
    # クラスごとのカウント
    class_counts = defaultdict(int)
    for td in traindata_list:
        label_name = td.label.label_name if td.label else 'unlabeled'
        class_counts[label_name] += 1
    
    total = sum(class_counts.values())
    
    # 割合を計算
    class_distribution = {}
    for label_name, count in sorted(class_counts.items()):
        percentage = (count / total * 100) if total > 0 else 0
        class_distribution[label_name] = {
            'count': count,
            'percentage': percentage
        }
    
    return class_distribution, total


def print_distribution(distribution, total, split_name="全体"):
    """クラス分布を表形式で表示"""
    print(f"\n{'='*60}")
    print(f"{split_name} (合計: {total}枚)")
    print(f"{'='*60}")
    print(f"{'クラス':<15} {'サンプル数':>10} {'割合':>10}")
    print(f"{'-'*60}")
    
    for label_name, info in sorted(distribution.items()):
        print(f"{label_name:<15} {info['count']:>10} {info['percentage']:>9.2f}%")
    print(f"{'='*60}")


def compare_distributions(overall_dist, split_dists):
    """
    全体の分布と各分割の分布を比較
    
    最大偏差を計算して層化分割が成功しているか判定
    """
    print(f"\n{'='*60}")
    print("クラス分布の比較（全体 vs 各分割）")
    print(f"{'='*60}")
    print(f"{'クラス':<10} {'全体%':>8} {'Train%':>8} {'Valid%':>8} {'Test%':>8} {'最大偏差':>10}")
    print(f"{'-'*60}")
    
    max_deviation = 0.0
    
    for label_name in sorted(overall_dist.keys()):
        overall_pct = overall_dist[label_name]['percentage']
        train_pct = split_dists['train'].get(label_name, {'percentage': 0})['percentage']
        valid_pct = split_dists['valid'].get(label_name, {'percentage': 0})['percentage']
        test_pct = split_dists['test'].get(label_name, {'percentage': 0})['percentage']
        
        # 最大偏差を計算
        deviations = [
            abs(train_pct - overall_pct),
            abs(valid_pct - overall_pct),
            abs(test_pct - overall_pct)
        ]
        max_dev = max(deviations)
        max_deviation = max(max_deviation, max_dev)
        
        print(f"{label_name:<10} {overall_pct:>7.2f}% {train_pct:>7.2f}% {valid_pct:>7.2f}% {test_pct:>7.2f}% {max_dev:>9.2f}%")
    
    print(f"{'='*60}")
    print(f"最大偏差: {max_deviation:.2f}%")
    
    if max_deviation < 5.0:
        print("✅ 層化分割成功！クラス分布がよく保たれています。")
    elif max_deviation < 10.0:
        print("⚠️  クラス分布にやや偏りがありますが、許容範囲内です。")
    else:
        print("❌ クラス分布に大きな偏りがあります。")
    
    return max_deviation


def main(theme_id: int, reset: bool = False):
    """
    メイン処理
    
    Args:
        theme_id: テーマID
        reset: Trueの場合、分割をリセットして再実行
    """
    print("============================================================")
    print("層化分割（Stratified Split）の動作確認")
    print("============================================================")
    
    theme = get_theme(theme_id)
    if not theme:
        print(f"❌ エラー: テーマID {theme_id} が見つかりません。")
        return
    
    print(f"\n✓ テーマ: {theme.name} (ID: {theme.id})")
    
    # 分割をリセット（オプション）
    if reset:
        print("\n分割をリセット中...")
        reset_splits(theme_id)
        print("✓ リセット完了")
    
    # 現在の統計を確認
    stats_before = get_split_statistics(theme_id)
    print(f"\n現在のデータ統計:")
    print(f"  Train: {stats_before.get('train', 0)}枚")
    print(f"  Valid: {stats_before.get('valid', 0)}枚")
    print(f"  Test: {stats_before.get('test', 0)}枚")
    print(f"  未分割: {stats_before.get('unsplit', 0)}枚")
    
    # 未分割データがある場合、分割を実行
    if stats_before.get('unsplit', 0) > 0:
        print("\n未分割データを分割中...")
        assign_splits_to_new_data(
            theme_id=theme_id,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        print("✓ 分割完了")
    
    # 分割後の統計
    stats_after = get_split_statistics(theme_id)
    
    # 全体のクラス分布を計算
    overall_dist, overall_total = calculate_class_distribution(theme_id)
    print_distribution(overall_dist, overall_total, "全体のクラス分布")
    
    # 各分割のクラス分布を計算
    split_dists = {}
    for split_name in ['train', 'valid', 'test']:
        dist, total = calculate_class_distribution(theme_id, split_name)
        split_dists[split_name] = dist
        print_distribution(dist, total, f"{split_name.upper()}のクラス分布")
    
    # 分布を比較
    max_deviation = compare_distributions(overall_dist, split_dists)
    
    print("\n============================================================")
    print("検証完了")
    print("============================================================")
    
    return max_deviation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="層化分割の動作確認スクリプト")
    parser.add_argument("--theme-id", type=int, default=7, help="確認するテーマのID")
    parser.add_argument("--reset", action="store_true", help="分割をリセットして再実行")
    args = parser.parse_args()
    
    main(args.theme_id, args.reset)

