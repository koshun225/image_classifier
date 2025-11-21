"""
MLflow連携

MLflowへのモデル登録、メトリクスとパラメータの記録、
params.yamlをMLflow artifactsとして保存、
Git commit IDとDVCバージョンの記録を行います。
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str = None, experiment_name: str = None):
    """
    MLflowのセットアップ
    
    Args:
        tracking_uri: MLflowのtracking URI
        experiment_name: 実験名
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")


def get_git_commit_id() -> Optional[str]:
    """
    現在のGit commit IDを取得
    
    Returns:
        Git commit ID（取得できない場合はNone）
    """
    try:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        logger.info(f"Git commit ID: {commit_id}")
        return commit_id
    except Exception as e:
        logger.warning(f"Git commit IDの取得に失敗: {e}")
        return None


def get_git_branch() -> Optional[str]:
    """
    現在のGitブランチ名を取得
    
    Returns:
        Gitブランチ名（取得できない場合はNone）
    """
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        logger.info(f"Git branch: {branch}")
        return branch
    except Exception as e:
        logger.warning(f"Gitブランチの取得に失敗: {e}")
        return None


def get_dvc_version(file_path: str) -> Optional[str]:
    """
    DVCファイルのバージョン（ハッシュ）を取得
    
    Args:
        file_path: DVCファイルのパス（例: "data.dvc"）
    
    Returns:
        DVCバージョン（ハッシュ）（取得できない場合はNone）
    """
    try:
        dvc_file = Path(file_path)
        if not dvc_file.exists():
            logger.warning(f"DVCファイルが見つかりません: {file_path}")
            return None
        
        with open(dvc_file, "r") as f:
            dvc_data = yaml.safe_load(f)
        
        # ハッシュを取得（md5またはouts[0].md5）
        if "md5" in dvc_data:
            dvc_hash = dvc_data["md5"]
        elif "outs" in dvc_data and len(dvc_data["outs"]) > 0:
            dvc_hash = dvc_data["outs"][0].get("md5")
        else:
            logger.warning(f"DVCハッシュが見つかりません: {file_path}")
            return None
        
        logger.info(f"DVC version ({file_path}): {dvc_hash}")
        return dvc_hash
    except Exception as e:
        logger.warning(f"DVCバージョンの取得に失敗: {e}")
        return None


def log_params_from_file(params_file: str):
    """
    params.yamlファイルからパラメータをMLflowにログ
    
    Args:
        params_file: params.yamlファイルのパス
    """
    try:
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)
        
        # フラット化してログ
        flat_params = flatten_dict(params)
        mlflow.log_params(flat_params)
        
        logger.info(f"パラメータをMLflowにログしました: {params_file}")
    except Exception as e:
        logger.error(f"パラメータのログに失敗: {e}")


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    ネストされた辞書をフラット化
    
    Args:
        d: 辞書
        parent_key: 親キー
        sep: セパレータ
    
    Returns:
        フラット化された辞書
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_artifacts_from_dir(artifacts_dir: str, artifact_path: str = None):
    """
    ディレクトリ内のファイルをMLflow artifactsとしてログ
    
    Args:
        artifacts_dir: ディレクトリのパス
        artifact_path: MLflow内のartifact path
    """
    try:
        mlflow.log_artifacts(artifacts_dir, artifact_path=artifact_path)
        logger.info(f"Artifactsをログしました: {artifacts_dir}")
    except Exception as e:
        logger.error(f"Artifactsのログに失敗: {e}")


def save_and_log_params(params: Dict[str, Any], save_path: str = "params_used.yaml"):
    """
    パラメータを保存してMLflowにログ
    
    Args:
        params: パラメータの辞書
        save_path: 保存先のパス
    """
    try:
        # YAMLファイルとして保存
        with open(save_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        
        # MLflowにartifactとしてログ
        mlflow.log_artifact(save_path, artifact_path="params")
        
        # パラメータをフラット化してログ
        flat_params = flatten_dict(params)
        mlflow.log_params(flat_params)
        
        logger.info(f"パラメータを保存してMLflowにログしました: {save_path}")
    except Exception as e:
        logger.error(f"パラメータの保存・ログに失敗: {e}")


def log_model_metadata(
    data_version: str = None,
    config_file: str = "config.yaml",
    data_dvc_file: str = "data.dvc"
):
    """
    モデルのメタデータをMLflowにログ
    
    Args:
        data_version: データバージョン（例: "ver1+ver2+ver3"）
        config_file: config.yamlファイルのパス
        data_dvc_file: data.dvcファイルのパス
    """
    # Git情報
    commit_id = get_git_commit_id()
    if commit_id:
        mlflow.log_param("git_commit_id", commit_id)
    
    branch = get_git_branch()
    if branch:
        mlflow.log_param("git_branch", branch)
    
    # DVCバージョン
    dvc_version = get_dvc_version(data_dvc_file)
    if dvc_version:
        mlflow.log_param("dvc_data_version", dvc_version)
    
    # データバージョン
    if data_version:
        mlflow.log_param("data_version", data_version)
    
    # config.yamlをログ
    if Path(config_file).exists():
        mlflow.log_artifact(config_file, artifact_path="config")
        logger.info(f"config.yamlをログしました: {config_file}")


def log_pytorch_model(
    model: torch.nn.Module,
    artifact_path: str = "model",
    registered_model_name: str = None,
    **kwargs
):
    """
    PyTorchモデルをMLflowにログ
    
    Args:
        model: PyTorchモデル
        artifact_path: MLflow内のartifact path
        registered_model_name: 登録するモデル名
        **kwargs: その他のmlflow.pytorch.log_modelのパラメータ
    """
    try:
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            **kwargs
        )
        logger.info(f"PyTorchモデルをMLflowにログしました: {artifact_path}")
    except Exception as e:
        logger.error(f"モデルのログに失敗: {e}")


def create_mlflow_run(
    experiment_name: str,
    run_name: str = None,
    tags: Dict[str, str] = None
) -> str:
    """
    MLflow runを作成
    
    Args:
        experiment_name: 実験名
        run_name: run名
        tags: タグの辞書
    
    Returns:
        run ID
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run作成: {run_id}")
        return run_id


def end_mlflow_run():
    """
    MLflow runを終了
    """
    mlflow.end_run()
    logger.info("MLflow run終了")


if __name__ == "__main__":
    # テスト
    print("MLflow Utils")
    
    # Git情報の取得
    commit_id = get_git_commit_id()
    branch = get_git_branch()
    print(f"Git commit ID: {commit_id}")
    print(f"Git branch: {branch}")
    
    # DVCバージョンの取得
    dvc_version = get_dvc_version("data.dvc")
    print(f"DVC version: {dvc_version}")
