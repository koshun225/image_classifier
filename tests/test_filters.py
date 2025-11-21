#!/usr/bin/env python
"""
複合フィルタ機能のテストスクリプト

ラベルと分割状態を組み合わせたフィルタリングが正しく動作するかテスト
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
from data_management.crud import get_theme, get_labels_by_theme


def test_filter(theme_id: int, filter_type: str = 'all', label_id: int = None, split: str = None):
    """
    フィルタ条件に基づいてデータを取得
    
    Args:
        theme_id: テーマID
        filter_type: 基本フィルタ（all, unlabeled, labeled）
        label_id: ラベルID（オプション）
        split: 分割（train, valid, test, unsplit）（オプション）
    
    Returns:
        フィルタ結果のQuerySet
    """
    queryset = TrainData.objects.filter(theme_id=theme_id)
    
    # 基本フィルタ
    if filter_type == 'unlabeled':
        queryset = queryset.filter(label__isnull=True)
    elif filter_type == 'labeled':
        queryset = queryset.filter(label__isnull=False)
    
    # ラベルフィルタ
    if label_id:
        queryset = queryset.filter(label_id=label_id)
    
    # 分割フィルタ
    if split:
        if split == 'unsplit':
            queryset = queryset.filter(split__isnull=True)
        elif split in ['train', 'valid', 'test']:
            queryset = queryset.filter(split=split)
    
    return queryset


def print_filter_result(theme_id: int, filter_type: str = 'all', label_id: int = None, 
                       split: str = None, label_name: str = None):
    """フィルタ結果を表示"""
    result = test_filter(theme_id, filter_type, label_id, split)
    count = result.count()
    
    # フィルタ条件の説明
    conditions = []
    if filter_type == 'unlabeled':
        conditions.append("未ラベル")
    elif filter_type == 'labeled':
        conditions.append("ラベル済み")
    else:
        conditions.append("すべて")
    
    if label_name:
        conditions.append(f"ラベル={label_name}")
    
    if split:
        conditions.append(f"分割={split}")
    
    condition_str = " AND ".join(conditions)
    
    print(f"  {condition_str}: {count}枚")
    return count


def main(theme_id: int):
    """メイン処理"""
    print("============================================================")
    print("複合フィルタ機能のテスト")
    print("============================================================")
    
    theme = get_theme(theme_id)
    if not theme:
        print(f"❌ エラー: テーマID {theme_id} が見つかりません。")
        return
    
    print(f"\n✓ テーマ: {theme.name} (ID: {theme.id})")
    
    labels = get_labels_by_theme(theme_id)
    print(f"✓ ラベル数: {len(labels)}個")
    print(f"  ラベル: {[label.label_name for label in labels]}")
    
    # テスト1: 基本フィルタのみ
    print("\n" + "="*60)
    print("テスト1: 基本フィルタ")
    print("="*60)
    
    total = print_filter_result(theme_id, 'all')
    unlabeled = print_filter_result(theme_id, 'unlabeled')
    labeled = print_filter_result(theme_id, 'labeled')
    
    if total == unlabeled + labeled:
        print("✅ 基本フィルタが正しく動作しています")
    else:
        print(f"❌ エラー: 合計が一致しません（{total} != {unlabeled} + {labeled}）")
    
    # テスト2: 分割フィルタ
    print("\n" + "="*60)
    print("テスト2: 分割フィルタ")
    print("="*60)
    
    train = print_filter_result(theme_id, 'all', split='train')
    valid = print_filter_result(theme_id, 'all', split='valid')
    test = print_filter_result(theme_id, 'all', split='test')
    unsplit = print_filter_result(theme_id, 'all', split='unsplit')
    
    if total == train + valid + test + unsplit:
        print("✅ 分割フィルタが正しく動作しています")
    else:
        print(f"❌ エラー: 合計が一致しません（{total} != {train} + {valid} + {test} + {unsplit}）")
    
    # テスト3: ラベルごとのフィルタ
    print("\n" + "="*60)
    print("テスト3: ラベルフィルタ")
    print("="*60)
    
    label_totals = []
    for label in labels:
        count = print_filter_result(theme_id, 'all', label_id=label.id, label_name=label.label_name)
        label_totals.append(count)
    
    if labeled == sum(label_totals):
        print("✅ ラベルフィルタが正しく動作しています")
    else:
        print(f"❌ エラー: ラベル済み合計が一致しません（{labeled} != {sum(label_totals)}）")
    
    # テスト4: 複合フィルタ（ラベル + 分割）
    print("\n" + "="*60)
    print("テスト4: 複合フィルタ（ラベル + 分割）")
    print("="*60)
    
    # 各ラベルのtrain/valid/test分布を確認
    for label in labels[:3]:  # 最初の3つのラベルで確認
        print(f"\nラベル '{label.label_name}' の分割:")
        label_train = print_filter_result(theme_id, 'all', label_id=label.id, split='train', label_name=label.label_name)
        label_valid = print_filter_result(theme_id, 'all', label_id=label.id, split='valid', label_name=label.label_name)
        label_test = print_filter_result(theme_id, 'all', label_id=label.id, split='test', label_name=label.label_name)
        label_unsplit = print_filter_result(theme_id, 'all', label_id=label.id, split='unsplit', label_name=label.label_name)
        label_total = print_filter_result(theme_id, 'all', label_id=label.id, label_name=label.label_name)
        
        if label_total == label_train + label_valid + label_test + label_unsplit:
            print(f"  ✅ ラベル '{label.label_name}' の分割が正しい")
        else:
            print(f"  ❌ ラベル '{label.label_name}' の分割に誤りがあります")
    
    print("\n" + "="*60)
    print("✅ すべてのテストが完了しました")
    print("="*60)
    
    # 使用例を表示
    print("\n" + "="*60)
    print("Django Web UIでの使用例")
    print("="*60)
    print(f"1. すべての画像: ?filter=all")
    print(f"2. 未ラベル画像: ?filter=unlabeled")
    print(f"3. ラベル3の画像: ?label={labels[2].id if len(labels) > 2 else 'X'}")
    print(f"4. Trainデータ: ?split=train")
    print(f"5. ラベル3のTrainデータ: ?label={labels[2].id if len(labels) > 2 else 'X'}&split=train")
    print(f"6. ラベル済みのValidデータ: ?filter=labeled&split=valid")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="複合フィルタ機能のテストスクリプト")
    parser.add_argument("--theme-id", type=int, default=7, help="テストするテーマのID")
    args = parser.parse_args()
    
    main(args.theme_id)

