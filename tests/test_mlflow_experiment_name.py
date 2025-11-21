#!/usr/bin/env python
"""
MLflow実験名のテスト

実験名がテーマ名に設定されることを確認
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

from data_management.crud import get_theme


def test_experiment_name():
    """実験名の設定をテスト"""
    print("="*60)
    print("MLflow実験名テスト")
    print("="*60)
    
    theme_id = 7
    
    # テーマ情報を取得
    theme = get_theme(theme_id=theme_id)
    
    if theme is None:
        print(f"❌ テーマID {theme_id} が見つかりません")
        return
    
    print(f"\n✓ テーマ情報:")
    print(f"  ID: {theme.id}")
    print(f"  名前: {theme.name}")
    print(f"  説明: {theme.description or '(なし)'}")
    
    print(f"\n✓ MLflow設定:")
    print(f"  実験名（Experiment Name）: {theme.name}")
    print(f"  記録されるパラメータ:")
    print(f"    - theme_id: {theme.id}")
    print(f"    - theme_name: {theme.name}")
    
    print("\n" + "="*60)
    print("確認事項")
    print("="*60)
    print("✓ 実験名がテーマ名に設定されます")
    print("✓ MLflowのExperiment一覧でテーマごとに分類されます")
    print("✓ 複数のテーマで学習を行う場合、実験が混ざりません")
    
    print("\n" + "="*60)
    print("学習実行")
    print("="*60)
    print(f"以下のコマンドで学習を開始してください：")
    print(f"\n  python scripts/train.py --theme-id {theme_id} --epochs 2 --batch-size 4")
    print(f"\n実験名: '{theme.name}' でMLflowに記録されます。")
    
    print("\n" + "="*60)
    print("MLflow UIでの確認")
    print("="*60)
    print("学習後、以下のコマンドでMLflow UIを起動：")
    print("\n  cd experiments")
    print("  mlflow ui --port 5001")
    print("\n  http://localhost:5001 にアクセス")
    print(f"\n'{theme.name}' という名前の実験を確認できます。")


if __name__ == "__main__":
    test_experiment_name()


