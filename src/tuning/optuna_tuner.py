"""
Optunaチューナー

params.yaml内のOptuna設定を読み込み、
Optuna Studyの作成と管理、
各トライアルでのparams.yaml動的生成、
トライアル結果のMLflow記録を行います。

親子ラン構造:
- 親ラン: params.yamlのrun_nameを使用（チューニング全体）
- 子ラン: 各トライアル（run_name=trial番号）
- モデル登録: best trialのみ
"""

import optuna
import yaml
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import copy
import os
import sys

from src.training.train import train as train_model
from src.utils.params_schema import (
    materialize_params,
    extract_tunable_specs,
    set_nested_value,
)

logger = logging.getLogger(__name__)


def _suggest_value(trial: optuna.Trial, path: str, spec: Dict[str, Any]):
    param_type = spec.get("type")
    if param_type is None:
        raise ValueError(f"{path}: type が指定されていません")

    if param_type == "float":
        return trial.suggest_float(
            path,
            spec["low"],
            spec["high"],
            log=spec.get("log", False),
        )
    if param_type == "int":
        return trial.suggest_int(
            path,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
        )
    if param_type == "categorical":
        choices = spec.get("choices")
        if not choices:
            raise ValueError(f"{path}: categorical には choices が必要です")
        return trial.suggest_categorical(path, choices)

    raise ValueError(f"サポートされていないパラメータタイプ: {param_type}")


def objective(
    trial: optuna.Trial,
    optuna_config: Dict[str, Any],
    parent_run_id: str,
    base_params: Dict[str, Any],
    tunable_specs: Dict[str, Dict[str, Any]],
    config_file: str = "config.yaml",
    augments_config: str = "auguments.yaml",
    **kwargs
) -> float:
    """
    Optunaの目的関数（親子ラン構造対応）
    
    Args:
        trial: Optunaトライアル
        optuna_config: Optuna設定
        parent_run_id: 親ランのMLflow Run ID
        config_file: config.yamlファイルのパス
        augments_config: auguments.yamlファイルのパス
        **kwargs: その他のパラメータ
    
    Returns:
        評価メトリクス
    """
    params = copy.deepcopy(base_params)
    for path, spec in tunable_specs.items():
        value = _suggest_value(trial, path, spec)
        set_nested_value(params, path.split("."), value)
    
    # params.yamlを一時ファイルとして保存
    temp_params_file = f"params_trial_{trial.number}.yaml"
    with open(temp_params_file, "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    
    logger.info(f"Trial {trial.number}: {temp_params_file} を生成しました")
    
    # 子ラン名をtrial番号に設定
    child_run_name = f"trial_{trial.number}"
    
    # 変数の初期化（スコープ対策）
    child_run_id = None
    test_results = None
    return_value = None
    
    try:
        # 親子ラン構造で学習を実行
        # 親ランは既にアクティブなので、直接子ランを作成
        with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
            child_run_id = child_run.info.run_id
            logger.info(f"Trial {trial.number}: 子ランID={child_run_id}")
            
            # トライアルパラメータをログ
            for param_name, param_value in trial.params.items():
                mlflow.log_param(param_name, param_value)
            
            mlflow.log_param("trial_number", trial.number)
            
            # 子ランのコンテキスト内で学習を実行
            # 既存の子ランIDを使ってMLFlowLoggerを初期化
            
            results = train_model(
                params_file=temp_params_file,
                config_file=config_file,
                augments_config=augments_config,
                enable_mlflow=True,  # MLflowLoggerを使用
                mlflow_run_id=child_run_id,  # 既存の子ランIDを使用
                run_name=child_run_name,
                **kwargs
            )
            
            # 評価メトリクスを取得
            test_results = results["test_results"]
            
            # 最大化するメトリクスを取得（例: test_acc）
            metric_name = optuna_config.get("metric", "test_acc")
            metric_value = test_results[0].get(metric_name)
            
            if metric_value is None:
                logger.error(f"メトリクス {metric_name} が見つかりません")
                raise ValueError(f"メトリクス {metric_name} が見つかりません")
            
            logger.info(f"Trial {trial.number}: {metric_name}={metric_value}")
            
            # objective_valueをログ（MLFlowLoggerが自動的にtrain/valid/testメトリクスをログ済み）
            mlflow.log_metric("objective_value", metric_value)
            
            # パラメータファイルをログ
            mlflow.log_artifact(temp_params_file, artifact_path="config")
            
            # 子ランのコンテキスト内でmetric_valueを返す準備
            return_value = metric_value
        
        # トライアルにメトリクスを記録
        trial.set_user_attr("test_results", test_results)
        trial.set_user_attr("mlflow_run_id", child_run_id)
        trial.set_user_attr("metric_value", return_value)
        
        return return_value
    
    except Exception as e:
        logger.error(f"Trial {trial.number} でエラーが発生しました: {e}", exc_info=True)
        raise
    
    finally:
        # 一時ファイルを削除
        if Path(temp_params_file).exists():
            Path(temp_params_file).unlink()


def tune(
    params_file: str = "params.yaml",
    config_file: str = "config.yaml",
    augments_config: str = "auguments.yaml",
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    load_if_exists: bool = True,
    training_job_id: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Optunaによるハイパーパラメータチューニング（親子ラン構造対応）
    
    親ランを作成し、各トライアルを子ランとして実行します。
    best trialのみモデルを登録します。
    
    Args:
        params_file: params.yamlファイルのパス（run_name取得用）
        config_file: config.yamlファイルのパス
        augments_config: auguments.yamlファイルのパス
        storage: Optunaのストレージ（例: "sqlite:///optuna.db"）
        study_name: Study名（Noneの場合は自動生成）
        load_if_exists: 既存のStudyをロードするか
        **kwargs: その他のパラメータ
    
    Returns:
        チューニング結果の辞書
    """
    with open(params_file, "r") as f:
        params_schema = yaml.safe_load(f) or {}
    
    optuna_config = params_schema.get("optuna", {}) or {}

    base_params = materialize_params(params_schema)
    tunable_specs = extract_tunable_specs(params_schema)

    training_config = base_params.get("training", {})
    parent_run_name = training_config.get("run_name", "optuna_tuning")

    data_config = base_params.get("data", {})
    theme_id = data_config.get("theme_id")
    if theme_id is None:
        raise ValueError("theme_idがparams.yamlのdataセクションに設定されていません")
    
    # Django環境のセットアップ
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / 'src' / 'web'))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    import django
    django.setup()
    
    from data_management.crud import get_theme
    from data_management.models import TrainingJob
    theme = get_theme(theme_id=theme_id)
    if theme is None:
        raise ValueError(f"テーマID {theme_id} が見つかりません")
    
    theme_name = theme.name
    logger.info(f"テーマ '{theme_name}' (ID: {theme_id}) でチューニングを開始します")
    
    # MLflowのセットアップ
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "experiments/mlruns")

    experiment_name = theme_name  # テーマ名を実験名として使用
    
    from src.utils.mlflow_utils import setup_mlflow
    setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    
    logger.info(f"MLflow実験名: '{experiment_name}'")
    logger.info(f"親ラン名: '{parent_run_name}'")
    
    # Study設定の取得
    if study_name is None:
        study_name = optuna_config.get("study_name") or f"{theme_name}_optuna"
    
    direction = optuna_config.get("direction", "maximize")
    n_trials = optuna_config.get("n_trials", 50)
    timeout = optuna_config.get("timeout")
    
    # Studyの作成
    logger.info(f"Optuna Study作成: {study_name}, direction={direction}")
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists
    )
    
    training_job = None
    if training_job_id is not None:
        try:
            training_job = TrainingJob.objects.get(id=training_job_id)
        except TrainingJob.DoesNotExist:
            logger.warning(f"TrainingJob(id={training_job_id}) が見つかりません")
            training_job = None

    # 親ランを作成
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"親ランID: {parent_run_id}")

        if training_job is not None:
            training_job.mlflow_parent_run_id = parent_run_id
            training_job.save(update_fields=["mlflow_parent_run_id"])
        
        # 親ランにチューニング設定をログ
        mlflow.log_param("theme_id", theme_id)
        mlflow.log_param("theme_name", theme_name)
        mlflow.log_param("study_name", study_name)
        mlflow.log_param("direction", direction)
        mlflow.log_param("n_trials", n_trials)
        
        # 設定ファイルをログ
        mlflow.log_artifact(params_file, artifact_path="config")
        mlflow.log_artifact(config_file, artifact_path="config")
        mlflow.log_artifact(augments_config, artifact_path="config")
        mlflow.log_dict(optuna_config, artifact_file="config/optuna.json")
        
        # 最適化の実行
        logger.info(f"最適化を開始します: n_trials={n_trials}, timeout={timeout}")
        def _objective_wrapper(trial):
            return objective(
                trial,
                optuna_config,
                parent_run_id=parent_run_id,
                base_params=base_params,
                tunable_specs=tunable_specs,
                config_file=config_file,
                augments_config=augments_config,
                **kwargs,
            )

        study.optimize(
            _objective_wrapper,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        # 最良のトライアル
        best_trial = study.best_trial
        logger.info(f"最良のトライアル: {best_trial.number}")
        logger.info(f"最良の値: {best_trial.value}")
        logger.info(f"最良のパラメータ: {best_trial.params}")
        
        # 親ランに最良の結果をログ
        mlflow.log_param("best_trial_number", best_trial.number)
        mlflow.log_metric("best_value", best_trial.value)
        
        # 最良のパラメータをログ
        for param_name, param_value in best_trial.params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        # 最良のパラメータでparams.yamlを生成
        best_params = copy.deepcopy(base_params)
        for param_path, value in best_trial.params.items():
            set_nested_value(best_params, param_path.split("."), value)
        
        # 最良のparams.yamlを保存
        best_params_file = "params_best.yaml"
        with open(best_params_file, "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)
        
        logger.info(f"最良のパラメータを保存しました: {best_params_file}")
        mlflow.log_artifact(best_params_file, artifact_path="best_params")
        
        # best trialのMLflow run IDを取得
        best_trial_run_id = best_trial.user_attrs.get("mlflow_run_id")
        if best_trial_run_id:
            logger.info(f"best trialのMLflow run ID: {best_trial_run_id}")
        
        # モデル登録は手動で行うため、ここでは実行しない
        logger.info("モデル登録は手動で実行してください")
        logger.info(f"  モデル登録コマンド例:")
        logger.info(f"  python scripts/register_model.py --run-id {best_trial_run_id} --model-name {theme_name}_model")
    
    # 結果
    results = {
        "study": study,
        "best_trial": best_trial,
        "best_params": best_params,
        "best_params_file": best_params_file,
        "best_value": best_trial.value,
        "parent_run_id": parent_run_id,
        "best_trial_run_id": best_trial_run_id,
    }
    
    logger.info("チューニングが完了しました")
    return results


if __name__ == "__main__":
    # テスト用
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s"
    )
    
    results = tune(
        config_file="config.yaml",
        augments_config="auguments.yaml"
    )
    
    print("チューニング結果:")
    print(f"  最良のトライアル: {results['best_trial'].number}")
    print(f"  最良の値: {results['best_value']}")
    print(f"  最良のパラメータ: {results['best_params']}")
    print(f"  最良のパラメータファイル: {results['best_params_file']}")
