"""
テスト用のテーマとラベルを本番データベースに作成するスクリプト

使い方:
    python scripts/create_test_theme.py
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src' / 'web'))

# Django環境をセットアップ
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from data_management.crud import (
    create_theme,
    create_label,
    get_all_themes,
    get_labels_by_theme
)


def create_mnist_test_theme():
    """MNIST Test テーマを作成"""
    
    print("=" * 60)
    print("MNIST Test テーマを作成します")
    print("=" * 60)
    
    # 既存のテーマを確認
    existing_themes = get_all_themes()
    for theme in existing_themes:
        if theme.name == "MNIST Test":
            print(f"\n⚠️  テーマ 'MNIST Test' は既に存在します（ID: {theme.id}）")
            response = input("削除して再作成しますか？ (y/n): ")
            if response.lower() == 'y':
                theme.delete()
                print("✓ 既存のテーマを削除しました")
            else:
                print("✓ 既存のテーマを使用します")
                return theme
    
    # テーマを作成
    theme = create_theme(
        name="MNIST Test",
        description="Test theme for MNIST dataset (0-9 digits)"
    )
    print(f"\n✓ テーマを作成しました: {theme.name} (ID: {theme.id})")
    
    # ラベルを作成（0-9）
    print("\nラベルを作成中...")
    labels = []
    for i in range(10):
        label = create_label(
            theme_id=theme.id,
            label_name=str(i)
        )
        labels.append(label)
        print(f"  ✓ ラベル '{label.label_name}' を作成 (ID: {label.id})")
    
    print(f"\n✓ {len(labels)}個のラベルを作成しました")
    
    return theme


def show_theme_info(theme):
    """テーマ情報を表示"""
    labels = get_labels_by_theme(theme_id=theme.id)
    
    print("\n" + "=" * 60)
    print("作成されたテーマ情報")
    print("=" * 60)
    print(f"テーマID: {theme.id}")
    print(f"テーマ名: {theme.name}")
    print(f"説明: {theme.description}")
    print(f"ラベル数: {len(labels)}個")
    print("\nラベル一覧:")
    for label in labels:
        print(f"  - ID: {label.id}, 名前: '{label.label_name}'")
    
    print("\n" + "=" * 60)
    print("次のステップ")
    print("=" * 60)
    print("1. Django管理画面にアクセス:")
    print("   cd /Users/koshun/projects/classification_with_mlops/src/web")
    print("   python manage.py runserver")
    print("   http://127.0.0.1:8000/admin/")
    print()
    print("2. Label-Studio風UIにアクセス:")
    print("   http://127.0.0.1:8000/")
    print()
    print("3. データセットをテスト:")
    print("   from src.data.dataset import ClassificationDataset, get_dataset_info")
    print(f"   info = get_dataset_info(theme_id={theme.id})")
    print("   print(info)")
    print()


def main():
    """メイン処理"""
    try:
        # テーマを作成
        theme = create_mnist_test_theme()
        
        # テーマ情報を表示
        show_theme_info(theme)
        
        print("✓ 完了しました！")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

