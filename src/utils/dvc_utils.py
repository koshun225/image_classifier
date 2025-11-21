"""
DVC連携

DVCデータバージョンの取得とデータセットのバージョン管理を行います。
"""

import subprocess
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_dvc_file_hash(dvc_file: str) -> Optional[str]:
    """
    DVCファイルのハッシュを取得
    
    Args:
        dvc_file: DVCファイルのパス（例: "data.dvc"）
    
    Returns:
        DVCハッシュ（取得できない場合はNone）
    """
    try:
        dvc_path = Path(dvc_file)
        if not dvc_path.exists():
            logger.warning(f"DVCファイルが見つかりません: {dvc_file}")
            return None
        
        with open(dvc_path, "r") as f:
            dvc_data = yaml.safe_load(f)
        
        # ハッシュを取得
        if "md5" in dvc_data:
            dvc_hash = dvc_data["md5"]
        elif "outs" in dvc_data and len(dvc_data["outs"]) > 0:
            dvc_hash = dvc_data["outs"][0].get("md5")
        else:
            logger.warning(f"DVCハッシュが見つかりません: {dvc_file}")
            return None
        
        logger.info(f"DVC hash ({dvc_file}): {dvc_hash}")
        return dvc_hash
    except Exception as e:
        logger.error(f"DVCハッシュの取得に失敗: {e}")
        return None


def check_dvc_status() -> bool:
    """
    DVCの状態を確認
    
    Returns:
        DVCが正常に動作している場合True
    """
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("DVC status: OK")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC statusエラー: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("DVCコマンドが見つかりません。DVCがインストールされているか確認してください。")
        return False


def pull_dvc_data(targets: List[str] = None) -> bool:
    """
    DVCデータをpull
    
    Args:
        targets: pullするターゲットのリスト（Noneの場合は全体）
    
    Returns:
        成功した場合True
    """
    try:
        cmd = ["dvc", "pull"]
        if targets:
            cmd.extend(targets)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("DVC pull completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC pullエラー: {e.stderr}")
        return False


def get_dvc_remote_info() -> Optional[Dict[str, Any]]:
    """
    DVCリモートの情報を取得
    
    Returns:
        DVCリモート情報の辞書（取得できない場合はNone）
    """
    try:
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        remotes = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    name, url = parts
                    remotes[name] = url
        
        logger.info(f"DVC remotes: {remotes}")
        return remotes
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC remote listエラー: {e.stderr}")
        return None


def get_data_version_from_config(config_file: str = "config.yaml") -> Optional[str]:
    """
    config.yamlからデータバージョンを取得
    
    Args:
        config_file: config.yamlファイルのパス
    
    Returns:
        データバージョン文字列（例: "ver1+ver2+ver3"）
    """
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        versions = config.get("data", {}).get("last_used", {}).get("versions", [])
        if versions:
            version_str = "+".join(versions)
            logger.info(f"Data version from config: {version_str}")
            return version_str
        else:
            logger.warning("config.yamlにバージョン情報が見つかりません")
            return None
    except Exception as e:
        logger.error(f"config.yamlの読み込みに失敗: {e}")
        return None


def validate_dvc_setup() -> bool:
    """
    DVC環境のセットアップを検証
    
    Returns:
        セットアップが正常な場合True
    """
    logger.info("DVC環境の検証を開始します...")
    
    # DVCコマンドの存在確認
    try:
        subprocess.run(
            ["dvc", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("✓ DVCコマンドが見つかりました")
    except FileNotFoundError:
        logger.error("✗ DVCコマンドが見つかりません")
        return False
    
    # .dvcディレクトリの存在確認
    if not Path(".dvc").exists():
        logger.error("✗ .dvcディレクトリが見つかりません。DVCが初期化されていない可能性があります。")
        return False
    logger.info("✓ .dvcディレクトリが見つかりました")
    
    # DVCリモートの確認
    remotes = get_dvc_remote_info()
    if remotes:
        logger.info(f"✓ DVCリモート: {remotes}")
    else:
        logger.warning("⚠ DVCリモートが設定されていません")
    
    # DVC status確認
    if check_dvc_status():
        logger.info("✓ DVC statusは正常です")
    else:
        logger.warning("⚠ DVC statusにエラーがあります")
    
    logger.info("DVC環境の検証完了")
    return True


if __name__ == "__main__":
    # テスト
    print("DVC Utils")
    
    # DVC環境の検証
    validate_dvc_setup()
    
    # DVCファイルハッシュの取得
    data_hash = get_dvc_file_hash("data.dvc")
    print(f"Data hash: {data_hash}")
    
    # config.yamlからバージョン取得
    data_version = get_data_version_from_config("config.yaml")
    print(f"Data version: {data_version}")
