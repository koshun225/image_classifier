#!/usr/bin/env python
"""
ラベルごとの詳細統計を表示するテストスクリプト
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

from data_management.models import TrainData
from data_management.crud import get_theme, get_labels_by_theme, get_split_statistics


def main(theme_id: int):
    """メイン処理"""
    print("============================================================")
    print("ラベルごとの詳細統計")
    print("============================================================")
    
    theme = get_theme(theme_id)
    if not theme:
        print(f"❌ エラー: テーマID {theme_id} が見つかりません。")
        return
    
    print(f"\n✓ テーマ: {theme.name} (ID: {theme.id})")
    
    labels = get_labels_by_theme(theme_id)
    print(f"✓ ラベル数: {len(labels)}個")
    
    # 全体統計
    stats = get_split_statistics(theme_id)
    
    # ラベルごとの詳細統計を収集
    label_stats = []
    for label in labels:
        train_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split='train').count()
        valid_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split='valid').count()
        test_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split='test').count()
        unsplit_count = TrainData.objects.filter(theme_id=theme_id, label_id=label.id, split__isnull=True).count()
        total_count = train_count + valid_count + test_count + unsplit_count
        
        label_stats.append({
            'label': label,
            'train': train_count,
            'valid': valid_count,
            'test': test_count,
            'unsplit': unsplit_count,
            'total': total_count,
        })
    
    # 表形式で表示
    print("\n" + "="*90)
    print(f"{'ラベル':<15} {'Train':>10} {'Valid':>10} {'Test':>10} {'未分割':>10} {'合計':>10}")
    print("="*90)
    
    for stat in label_stats:
        print(f"{stat['label'].label_name:<15} "
              f"{stat['train']:>10} "
              f"{stat['valid']:>10} "
              f"{stat['test']:>10} "
              f"{stat['unsplit']:>10} "
              f"{stat['total']:>10}")
    
    print("-"*90)
    print(f"{'合計':<15} "
          f"{stats['train']:>10} "
          f"{stats['valid']:>10} "
          f"{stats['test']:>10} "
          f"{stats['unsplit']:>10} "
          f"{stats['train'] + stats['valid'] + stats['test'] + stats['unsplit']:>10}")
    print("="*90)
    
    # 比率を計算
    print("\n" + "="*90)
    print("各ラベルの分割比率（%）")
    print("="*90)
    print(f"{'ラベル':<15} {'Train%':>10} {'Valid%':>10} {'Test%':>10} {'未分割%':>10}")
    print("="*90)
    
    for stat in label_stats:
        total = stat['total']
        if total > 0:
            train_pct = (stat['train'] / total * 100)
            valid_pct = (stat['valid'] / total * 100)
            test_pct = (stat['test'] / total * 100)
            unsplit_pct = (stat['unsplit'] / total * 100)
            
            print(f"{stat['label'].label_name:<15} "
                  f"{train_pct:>9.1f}% "
                  f"{valid_pct:>9.1f}% "
                  f"{test_pct:>9.1f}% "
                  f"{unsplit_pct:>9.1f}%")
    
    total_all = stats['train'] + stats['valid'] + stats['test'] + stats['unsplit']
    if total_all > 0:
        print("-"*90)
        print(f"{'全体平均':<15} "
              f"{stats['train'] / total_all * 100:>9.1f}% "
              f"{stats['valid'] / total_all * 100:>9.1f}% "
              f"{stats['test'] / total_all * 100:>9.1f}% "
              f"{stats['unsplit'] / total_all * 100:>9.1f}%")
    print("="*90)
    
    # データバランスの検証
    print("\n" + "="*90)
    print("データバランスの検証")
    print("="*90)
    
    # 各ラベルのサンプル数が均等か確認
    totals = [stat['total'] for stat in label_stats]
    if totals:
        min_count = min(totals)
        max_count = max(totals)
        avg_count = sum(totals) / len(totals)
        
        print(f"最小サンプル数: {min_count}枚")
        print(f"最大サンプル数: {max_count}枚")
        print(f"平均サンプル数: {avg_count:.1f}枚")
        print(f"標準偏差: {(sum((t - avg_count)**2 for t in totals) / len(totals))**0.5:.2f}枚")
        
        if max_count - min_count <= 5:
            print("✅ ラベル間のデータバランスは非常に良好です")
        elif max_count - min_count <= 10:
            print("⚠️  ラベル間に若干の偏りがありますが、許容範囲内です")
        else:
            print("❌ ラベル間に大きな偏りがあります。データ追加を検討してください")
    
    # 層化分割の検証
    if stats['unsplit'] == 0:
        print("\n層化分割の検証:")
        train_ratios = [stat['train'] / stat['total'] if stat['total'] > 0 else 0 for stat in label_stats]
        valid_ratios = [stat['valid'] / stat['total'] if stat['total'] > 0 else 0 for stat in label_stats]
        test_ratios = [stat['test'] / stat['total'] if stat['total'] > 0 else 0 for stat in label_stats]
        
        train_avg = sum(train_ratios) / len(train_ratios) if train_ratios else 0
        valid_avg = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0
        test_avg = sum(test_ratios) / len(test_ratios) if test_ratios else 0
        
        train_std = (sum((r - train_avg)**2 for r in train_ratios) / len(train_ratios))**0.5 if train_ratios else 0
        valid_std = (sum((r - valid_avg)**2 for r in valid_ratios) / len(valid_ratios))**0.5 if valid_ratios else 0
        test_std = (sum((r - test_avg)**2 for r in test_ratios) / len(test_ratios))**0.5 if test_ratios else 0
        
        print(f"Train比率の標準偏差: {train_std:.4f} (平均: {train_avg:.2%})")
        print(f"Valid比率の標準偏差: {valid_std:.4f} (平均: {valid_avg:.2%})")
        print(f"Test比率の標準偏差: {test_std:.4f} (平均: {test_avg:.2%})")
        
        if max(train_std, valid_std, test_std) < 0.01:
            print("✅ 層化分割が完璧に実行されています")
        elif max(train_std, valid_std, test_std) < 0.05:
            print("✅ 層化分割が正常に実行されています")
        else:
            print("⚠️  分割比率に若干のばらつきがあります")
    
    print("\n✓ 統計表示完了！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ラベルごとの詳細統計を表示")
    parser.add_argument("--theme-id", type=int, default=7, help="テーマID")
    args = parser.parse_args()
    
    main(args.theme_id)

