#!/usr/bin/env python3
"""
DVCの追跡管理スクリプト

config.yamlのdvc_managed設定に基づいて、DVCの追跡を自動化します。
"""

import subprocess
import sys
from pathlib import Path
import yaml


def run_command(cmd, check=True):
    """
    コマンドを実行
    
    Args:
        cmd: 実行するコマンド（リスト形式）
        check: エラー時に例外を発生させるか
    
    Returns:
        subprocess.CompletedProcess
    """
    print(f"実行中: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0 and check:
        print(f"エラー: {result.stderr}")
        sys.exit(1)
    
    if result.stdout:
        print(result.stdout)
    
    return result


def is_dvc_initialized():
    """DVCが初期化されているかチェック"""
    return Path(".dvc").exists()


def is_tracked_by_dvc(path):
    """
    指定されたパスがDVCで追跡されているかチェック
    
    Args:
        path: チェックするパス
    
    Returns:
        bool: 追跡されている場合True
    """
    dvc_file = Path(f"{path}.dvc")
    return dvc_file.exists()


def add_to_dvc(path):
    """
    パスをDVCに追加
    
    Args:
        path: 追加するパス
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        print(f"警告: {path}は存在しません。スキップします。")
        return False
    
    if is_tracked_by_dvc(path):
        print(f"情報: {path}は既にDVCで追跡されています。")
        return True
    
    print(f"\n{path}をDVCに追加中...")
    result = run_command(["dvc", "add", path], check=False)
    
    if result.returncode == 0:
        print(f"✓ {path}をDVCに追加しました。")
        
        # .dvcファイルをGitに追加
        dvc_file = f"{path}.dvc"
        if Path(dvc_file).exists():
            run_command(["git", "add", dvc_file, ".gitignore"], check=False)
            print(f"✓ {dvc_file}をGitに追加しました。")
        
        return True
    else:
        print(f"✗ {path}のDVC追加に失敗しました。")
        return False


def remove_from_dvc(path):
    """
    パスをDVCの追跡から削除
    
    Args:
        path: 削除するパス
    """
    if not is_tracked_by_dvc(path):
        print(f"情報: {path}はDVCで追跡されていません。")
        return True
    
    print(f"\n{path}をDVCの追跡から削除中...")
    
    # dvc untracを実行（DVC 2.x以降）
    result = run_command(["dvc", "remove", f"{path}.dvc"], check=False)
    
    if result.returncode == 0:
        print(f"✓ {path}をDVCの追跡から削除しました。")
        
        # .dvcファイルをGitから削除
        dvc_file = f"{path}.dvc"
        if Path(dvc_file).exists():
            run_command(["git", "rm", dvc_file], check=False)
        
        return True
    else:
        print(f"✗ {path}のDVC削除に失敗しました。")
        return False


def sync_dvc_tracking(config_file="config.yaml"):
    """
    config.yamlの設定に基づいてDVCの追跡を同期
    
    Args:
        config_file: 設定ファイルのパス
    """
    print("=" * 60)
    print("DVC追跡管理スクリプト")
    print("=" * 60)
    
    # DVCの初期化チェック
    if not is_dvc_initialized():
        print("\nエラー: DVCが初期化されていません。")
        print("以下のコマンドを実行してください:")
        print("  dvc init")
        sys.exit(1)
    
    # config.yamlを読み込み
    print(f"\n{config_file}を読み込み中...")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # dvc_managed設定を取得
    dvc_managed = config.get("data", {}).get("dvc_managed", [])
    
    if not dvc_managed:
        print("警告: config.yamlにdvc_managed設定がありません。")
        return
    
    print(f"DVC管理対象: {', '.join(dvc_managed)}")
    
    # 各パスをDVCに追加
    print("\n" + "=" * 60)
    print("DVC追跡の同期")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for path in dvc_managed:
        result = add_to_dvc(path)
        if result is True:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            fail_count += 1
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("結果")
    print("=" * 60)
    print(f"成功: {success_count}個")
    print(f"スキップ: {skip_count}個")
    print(f"失敗: {fail_count}個")
    
    if fail_count == 0:
        print("\n✓ すべてのパスがDVCで追跡されています。")
    else:
        print(f"\n✗ {fail_count}個のパスで問題が発生しました。")
    
    # Gitコミットの提案
    print("\n" + "=" * 60)
    print("次のステップ")
    print("=" * 60)
    print("以下のコマンドで.dvcファイルをコミットしてください:")
    print("  git add *.dvc .gitignore")
    print("  git commit -m 'Add data tracking with DVC'")


def list_dvc_tracked():
    """DVCで追跡されているファイルを一覧表示"""
    print("=" * 60)
    print("DVC追跡ファイル一覧")
    print("=" * 60)
    
    dvc_files = list(Path(".").glob("**/*.dvc"))
    
    if not dvc_files:
        print("DVCで追跡されているファイルはありません。")
        return
    
    for dvc_file in sorted(dvc_files):
        tracked_path = str(dvc_file).replace(".dvc", "")
        status = "✓ 存在" if Path(tracked_path).exists() else "✗ 不在"
        print(f"  {tracked_path} [{status}]")
    
    print(f"\n合計: {len(dvc_files)}個のファイル/ディレクトリ")


def dvc_commit():
    """
    すべての.dvcファイルをコミット
    """
    print("=" * 60)
    print("DVC Commit")
    print("=" * 60)
    
    dvc_files = list(Path(".").glob("**/*.dvc"))
    
    if not dvc_files:
        print("DVCファイルが見つかりません。")
        return True
    
    print(f"\n{len(dvc_files)}個の.dvcファイルをコミット中...")
    
    for dvc_file in dvc_files:
        print(f"\n処理中: {dvc_file}")
        result = run_command(["dvc", "commit", str(dvc_file)], check=False)
        
        if result.returncode == 0:
            print(f"✓ {dvc_file} をコミットしました。")
        else:
            print(f"✗ {dvc_file} のコミットに失敗しました。")
    
    print("\n✓ すべての.dvcファイルをコミットしました。")
    return True


def dvc_push(remote=None):
    """
    DVCキャッシュをリモートストレージにプッシュ
    
    Args:
        remote: リモート名（指定しない場合はデフォルトリモート）
    """
    print("=" * 60)
    print("DVC Push")
    print("=" * 60)
    
    # リモート設定の確認
    result = run_command(["dvc", "remote", "list"], check=False)
    
    if result.returncode != 0 or not result.stdout.strip():
        print("\n警告: DVCリモートが設定されていません。")
        print("以下のコマンドでリモートを設定してください:")
        print("  dvc remote add -d myremote /path/to/remote")
        print("  dvc remote add -d myremote s3://mybucket/path")
        return False
    
    print(f"\n設定されているリモート:")
    print(result.stdout)
    
    # プッシュ実行
    print("\nDVCキャッシュをリモートにプッシュ中...")
    
    cmd = ["dvc", "push"]
    if remote:
        cmd.extend(["-r", remote])
    
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print("\n✓ DVCプッシュが完了しました。")
        return True
    else:
        print("\n✗ DVCプッシュに失敗しました。")
        return False


def full_sync(config_file="config.yaml", push=False, remote=None):
    """
    完全同期: add -> commit -> (push)
    
    Args:
        config_file: 設定ファイルのパス
        push: プッシュも実行するか
        remote: リモート名
    """
    print("=" * 60)
    print("DVC完全同期")
    print("=" * 60)
    
    # 1. 追跡の同期
    print("\nステップ1: 追跡の同期")
    sync_dvc_tracking(config_file)
    
    # 2. コミット
    print("\nステップ2: DVCコミット")
    dvc_commit()
    
    # 3. プッシュ（オプション）
    if push:
        print("\nステップ3: DVCプッシュ")
        dvc_push(remote)
    else:
        print("\nステップ3: DVCプッシュ（スキップ）")
        print("プッシュするには --push オプションを使用してください。")
    
    # 最終メッセージ
    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. Gitにコミット:")
    print("   git add *.dvc .gitignore")
    print("   git commit -m 'Update DVC tracking'")
    print("   git push")
    
    if not push:
        print("\n2. DVCプッシュ（手動実行）:")
        print("   dvc push")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="config.yamlの設定に基づいてDVCの追跡を管理します"
    )
    parser.add_argument(
        "action",
        choices=["sync", "list", "commit", "push", "full"],
        help=(
            "実行するアクション\n"
            "  sync: 追跡を同期 (dvc add)\n"
            "  list: 追跡ファイル一覧表示\n"
            "  commit: .dvcファイルをコミット (dvc commit)\n"
            "  push: リモートにプッシュ (dvc push)\n"
            "  full: 完全同期 (sync + commit + push)"
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルのパス（デフォルト: config.yaml）"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="fullアクション時にプッシュも実行"
    )
    parser.add_argument(
        "--remote",
        help="リモート名（指定しない場合はデフォルトリモート）"
    )
    
    args = parser.parse_args()
    
    if args.action == "sync":
        sync_dvc_tracking(args.config)
    elif args.action == "list":
        list_dvc_tracked()
    elif args.action == "commit":
        dvc_commit()
    elif args.action == "push":
        dvc_push(args.remote)
    elif args.action == "full":
        full_sync(args.config, push=args.push, remote=args.remote)


if __name__ == "__main__":
    main()

