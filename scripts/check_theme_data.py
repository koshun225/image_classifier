"""
Django CRUDを使用してテーマデータを確認

使い方:
    python scripts/check_theme_data.py --theme-id 7
"""

import os
import sys
from pathlib import Path
import argparse

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src' / 'web'))

# Django環境をセットアップ
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from data_management.crud import (
    get_theme,
    get_labels_by_theme,
    get_class_names_by_theme,
    get_split_statistics,
    get_traindata_by_theme
)


def check_theme_data(theme_id: int):
    """テーマデータを確認"""
    
    print("=" * 60)
    print(f"テーマID {theme_id} のデータを確認")
    print("=" * 60)
    
    try:
        # テーマを取得
        theme = get_theme(theme_id=theme_id)
        if theme is None:
            print(f"\n❌ テーマID {theme_id} が見つかりません")
            return False
        
        print(f"\n✓ テーマ情報:")
        print(f"  - ID: {theme.id}")
        print(f"  - 名前: {theme.name}")
        print(f"  - 説明: {theme.description}")
        print(f"  - 作成日時: {theme.created_at}")
        
        # ラベルを取得
        labels = get_labels_by_theme(theme_id=theme_id)
        class_names = get_class_names_by_theme(theme_id=theme_id)
        
        print(f"\n✓ ラベル情報:")
        print(f"  - ラベル数: {len(labels)}個")
        print(f"  - クラス名: {class_names}")
        
        # 分割統計を取得
        stats = get_split_statistics(theme_id=theme_id)
        
        print(f"\n✓ データ分割統計:")
        print(f"  - Train: {stats['train']}枚")
        print(f"  - Valid: {stats['valid']}枚")
        print(f"  - Test: {stats['test']}枚")
        print(f"  - 未分割: {stats['unsplit']}枚")
        
        total = stats['train'] + stats['valid'] + stats['test'] + stats['unsplit']
        print(f"  - 合計: {total}枚")
        
        # 学習データの詳細を取得（最初の数件のみ）
        if total > 0:
            print(f"\n✓ 学習データのサンプル（最初の5件）:")
            all_data = get_traindata_by_theme(theme_id=theme_id)
            for i, data in enumerate(all_data[:5]):
                label_name = data.label.label_name if data.label else "未ラベル"
                split_name = data.split if data.split else "未分割"
                print(f"  {i+1}. ID={data.id}, ラベル='{label_name}', 分割={split_name}, 画像={Path(data.image.name).name}")
        else:
            print(f"\n⚠️  画像データがまだ登録されていません")
        
        # 次のステップを表示
        print("\n" + "=" * 60)
        print("次のステップ")
        print("=" * 60)
        
        if total == 0:
            print("1. Django Web UIにアクセスして画像をアップロード:")
            print("   cd /Users/koshun/projects/classification_with_mlops/src/web")
            print("   python manage.py runserver")
            print("   http://127.0.0.1:8000/")
            print()
            print("2. テーマ一覧から 'MNIST Test' をクリック")
            print("3. 画像をアップロードしてラベル付け")
            print("4. データ分割を実行")
        else:
            print("✓ データが登録されています！")
            print()
            print("PyTorchをインストールしてデータセットをテスト:")
            print("  pip install torch torchvision")
            print()
            print("その後、以下のコードでデータセットを読み込めます:")
            print(f"  from src.data.dataset import ClassificationDataset")
            print(f"  dataset = ClassificationDataset(theme_id={theme_id}, split='train')")
            print(f"  print(f'データセットサイズ: {{len(dataset)}}枚')")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="テーマデータ確認")
    parser.add_argument("--theme-id", type=int, default=7, help="テーマID（デフォルト: 7）")
    
    args = parser.parse_args()
    
    success = check_theme_data(args.theme_id)
    
    if success:
        print("\n✓ チェック完了！")
    else:
        print("\n❌ チェック失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()

