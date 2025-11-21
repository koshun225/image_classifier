#!/usr/bin/env python
"""
Visualization Scripts 統合テスト

Django統合されたvisualizationスクリプトの動作を確認します。
"""

import os
import sys
from pathlib import Path
import argparse

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Django環境のセットアップ
sys.path.insert(0, str(project_root / 'src' / 'web'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from data_management.crud import get_theme, get_traindata_by_theme


def check_theme_data(theme_id: int):
    """テーマのデータ状況を確認"""
    print("="*70)
    print("テーマデータ確認")
    print("="*70)
    
    # テーマ取得
    theme = get_theme(theme_id=theme_id)
    if theme is None:
        print(f"❌ テーマID {theme_id} が見つかりません")
        return False
    
    print(f"\n✅ テーマが見つかりました:")
    print(f"  ID: {theme.id}")
    print(f"  名前: {theme.name}")
    print(f"  説明: {theme.description or '(なし)'}")
    
    # 全データ取得
    all_data = get_traindata_by_theme(theme_id=theme_id)
    if not all_data:
        print(f"\n❌ テーマID {theme_id} に画像が登録されていません")
        return False
    
    print(f"\n✅ 登録画像数: {len(all_data)}枚")
    
    # 分割データ確認
    train_data = get_traindata_by_theme(theme_id=theme_id, split="train")
    valid_data = get_traindata_by_theme(theme_id=theme_id, split="valid")
    test_data = get_traindata_by_theme(theme_id=theme_id, split="test")
    unsplit_data = get_traindata_by_theme(theme_id=theme_id, split=None)
    
    print(f"\nデータ分割状況:")
    print(f"  Train: {len(train_data)}枚")
    print(f"  Valid: {len(valid_data)}枚")
    print(f"  Test: {len(test_data)}枚")
    print(f"  Unsplit: {len(unsplit_data)}枚")
    
    if not train_data:
        print(f"\n⚠️  警告: trainデータがありません")
        print(f"  Django Web UIまたはスクリプトでデータ分割を実行してください")
        return False
    
    print(f"\n✅ データ分割が完了しています")
    
    # ラベル情報
    label_names = set(data.label.label_name for data in all_data)
    print(f"\nラベル数: {len(label_names)}")
    print(f"ラベル: {', '.join(sorted(label_names))}")
    
    return True


def check_augments_yaml():
    """auguments.yamlの存在確認"""
    print("\n" + "="*70)
    print("auguments.yaml 確認")
    print("="*70)
    
    augments_file = project_root / "auguments.yaml"
    
    if not augments_file.exists():
        print(f"\n❌ {augments_file} が見つかりません")
        return False
    
    print(f"\n✅ {augments_file} が存在します")
    
    # 内容の簡易チェック
    import yaml
    try:
        with open(augments_file, "r") as f:
            config = yaml.safe_load(f)
        
        print(f"\n設定セクション:")
        for key in config.keys():
            print(f"  - {key}")
        
        return True
    except Exception as e:
        print(f"\n❌ {augments_file} の読み込みエラー: {e}")
        return False


def test_visualization_scripts(theme_id: int, run_all: bool = False):
    """visualizationスクリプトの動作テスト"""
    print("\n" + "="*70)
    print("Visualization Scripts テスト")
    print("="*70)
    
    if run_all:
        print("\n注意: すべてのスクリプトを実行します（時間がかかります）")
        response = input("続行しますか？ [y/N]: ")
        if response.lower() != 'y':
            print("キャンセルしました")
            return
    
    scripts = [
        ("vis_preprocessing.py", f"--theme-id {theme_id}"),
        ("vis_augmentation.py", f"--theme-id {theme_id} --num-samples 4"),
        ("vis_dataset.py", f"--theme-id {theme_id} --num-samples 8"),
    ]
    
    print(f"\n実行するスクリプト:")
    for script, args in scripts:
        print(f"  - {script} {args}")
    
    if not run_all:
        print(f"\n💡 ヒント: 実際に実行するには --run-all オプションを付けてください")
        print(f"\n使用例:")
        for script, args in scripts:
            print(f"  python scripts/visualization/{script} {args}")
        return
    
    # 実行
    import subprocess
    
    results = []
    for script, args in scripts:
        script_path = project_root / "scripts" / "visualization" / script
        cmd = f"python {script_path} {args}"
        
        print(f"\n{'='*70}")
        print(f"実行中: {script}")
        print(f"コマンド: {cmd}")
        print(f"{'='*70}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5分タイムアウト
            )
            
            success = result.returncode == 0
            results.append((script, success))
            
            if success:
                print(f"✅ {script} 成功")
            else:
                print(f"❌ {script} 失敗")
                print(f"エラー出力:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"❌ {script} タイムアウト（5分超過）")
            results.append((script, False))
        except Exception as e:
            print(f"❌ {script} 実行エラー: {e}")
            results.append((script, False))
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("テスト結果サマリー")
    print(f"{'='*70}")
    
    for script, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {script}: {status}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print(f"\n🎉 すべてのテストが成功しました！")
        print(f"\n生成されたファイル:")
        workspace = project_root / "workspace"
        for file in sorted(workspace.glob("demo_*.png")):
            print(f"  - {file}")
    else:
        print(f"\n⚠️  一部のテストが失敗しました")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Visualization Scripts 統合テスト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme-id",
        type=int,
        required=True,
        help="テーマID"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="すべてのvisualizationスクリプトを実行"
    )
    args = parser.parse_args()
    
    print("="*70)
    print("Visualization Scripts 統合テスト")
    print("="*70)
    print(f"\nテーマID: {args.theme_id}")
    
    # 1. テーマデータ確認
    if not check_theme_data(args.theme_id):
        print(f"\n❌ テーマデータの確認に失敗しました")
        return 1
    
    # 2. auguments.yaml確認
    if not check_augments_yaml():
        print(f"\n❌ auguments.yamlの確認に失敗しました")
        return 1
    
    # 3. スクリプトテスト
    test_visualization_scripts(args.theme_id, run_all=args.run_all)
    
    print(f"\n{'='*70}")
    print("テスト完了")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

