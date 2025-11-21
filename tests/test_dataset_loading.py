"""
Djangoデータベースからのデータセット読み込みをテスト

使い方:
    python scripts/test_dataset_loading.py --theme-id 7
"""

import os
import sys
from pathlib import Path
import argparse

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import get_dataset_info
from src.data.split import get_split_info


def test_dataset_info(theme_id: int):
    """データセット情報を取得してテスト"""
    
    print("=" * 60)
    print(f"テーマID {theme_id} のデータセット情報を取得")
    print("=" * 60)
    
    try:
        # データセット情報を取得
        info = get_dataset_info(theme_id=theme_id)
        
        print("\n✓ データセット情報取得成功:")
        print(f"  - テーマID: {info['theme_id']}")
        print(f"  - テーマ名: {info['theme_name']}")
        print(f"  - 説明: {info['theme_description']}")
        print(f"  - クラス数: {info['num_classes']}")
        print(f"  - クラス名: {info['class_names']}")
        print(f"  - 総サンプル数: {info['total_samples']}")
        
        # 分割統計を取得
        split_stats = info['samples_per_split']
        print(f"\n✓ データ分割統計:")
        print(f"  - Train: {split_stats['train']}枚")
        print(f"  - Valid: {split_stats['valid']}枚")
        print(f"  - Test: {split_stats['test']}枚")
        print(f"  - 未分割: {split_stats['unsplit']}枚")
        
        if info['total_samples'] == 0:
            print("\n⚠️  画像データがまだ登録されていません")
            print("\n次のステップ:")
            print("1. Django Web UIにアクセス:")
            print("   cd /Users/koshun/projects/classification_with_mlops/src/web")
            print("   python manage.py runserver")
            print("   http://127.0.0.1:8000/")
            print()
            print("2. テーマ一覧から 'MNIST Test' をクリック")
            print("3. 画像をアップロードしてラベル付け")
            print("4. データ分割を実行")
            print("5. 再度このスクリプトを実行してデータセットを確認")
        else:
            print("\n✓ 画像データが登録されています！")
            print(f"\n次のステップ:")
            print(f"  - データセットを使用:")
            print(f"    from src.data.dataset import ClassificationDataset")
            print(f"    dataset = ClassificationDataset(theme_id={theme_id}, split='train')")
            print(f"    print(f'データセットサイズ: {{len(dataset)}}枚')")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="データセット読み込みテスト")
    parser.add_argument("--theme-id", type=int, default=7, help="テーマID（デフォルト: 7）")
    
    args = parser.parse_args()
    
    success = test_dataset_info(args.theme_id)
    
    if success:
        print("\n✓ テスト成功！")
    else:
        print("\n❌ テスト失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()

