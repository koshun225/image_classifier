#!/usr/bin/env python3
"""
ハイパーパラメータチューニング実行スクリプト

Optunaを使用してハイパーパラメータチューニングを実行します。

使用例:
    python scripts/tune.py
    python scripts/tune.py --n-trials 100 --timeout 3600
    python scripts/tune.py --storage sqlite:///optuna.db --study-name my_study
"""

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tuning.optuna_tuner import tune


def parse_args():
    """
    コマンドライン引数をパース
    
    Returns:
        argparse.Namespace: パースされた引数
    """
    parser = argparse.ArgumentParser(
        description="Optunaによるハイパーパラメータチューニング",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 設定ファイル
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="params.yamlファイルのパス（run_name、theme_id取得用）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="config.yamlファイルのパス"
    )
    parser.add_argument(
        "--augments",
        type=str,
        default="auguments.yaml",
        help="auguments.yamlファイルのパス"
    )
    
    # データ設定
    parser.add_argument(
        "--theme-id",
        type=int,
        default=None,
        help="テーマID（指定した場合、params.yamlのtheme_idを上書き）"
    )
    parser.add_argument(
        "--training-job-id",
        type=int,
        default=None,
        help="TrainingJob ID（Djangoデータベース連携用）"
    )
    
    # Optuna設定
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optunaストレージ（例: sqlite:///optuna.db）"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study名（指定しない場合は自動生成）"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="トライアル数（params.yaml内の設定を上書き）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="タイムアウト（秒）（params.yaml内の設定を上書き）"
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        default=True,
        help="既存のStudyをロードする"
    )
    
    # トレーニング設定
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "tpu", "mps"],
        help="使用するアクセラレータ"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="使用するデバイス（例: '1', '0,1', 'auto'）"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="精度（32bit, 16bit mixed precision, bfloat16）"
    )
    
    # その他
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベル"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_tuning",
        help="チェックポイントの保存ディレクトリ"
    )
    
    return parser.parse_args()


def setup_logging(log_level: str = "INFO"):
    """
    ロギングのセットアップ
    
    Args:
        log_level: ログレベル
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def override_params(params_file: str, args: argparse.Namespace):
    """
    params.yamlをコマンドライン引数で上書き
    
    Args:
        params_file: params.yamlファイルのパス
        args: コマンドライン引数
    
    Returns:
        上書きされたparams.yamlファイルのパス（上書きがない場合は元のパス）
    """
    import yaml
    
    import yaml
    
    # params.yamlを読み込み
    with open(params_file, "r") as f:
        params = yaml.safe_load(f) or {}
    
    modified = False
    
    if args.theme_id is not None:
        params.setdefault("data", {})["theme_id"] = args.theme_id
        modified = True
        logging.info(f"theme_idを上書き: {args.theme_id}")
    
    if args.n_trials is not None:
        params.setdefault("optuna", {})["n_trials"] = args.n_trials
        modified = True
        logging.info(f"n_trialsを上書き: {args.n_trials}")
    
    if args.timeout is not None:
        params.setdefault("optuna", {})["timeout"] = args.timeout
        modified = True
        logging.info(f"timeoutを上書き: {args.timeout}")
    
    if not modified:
        return params_file
    
    # 一時ファイルに保存
    temp_params_file = f"{params_file}.tmp"
    with open(temp_params_file, "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    logging.info(f"上書きされたparamsを保存: {temp_params_file}")
    
    return temp_params_file


def setup_django():
    """Django環境を初期化"""
    import os
    import django
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / 'src' / 'web'))
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()


def save_model_to_django(mlflow_run_id: str, theme_id: int, training_job_id: int, best_score: float, training_params: dict):
    """
    DjangoデータベースにModelレコードを作成
    
    Args:
        mlflow_run_id: MLflow Run ID
        theme_id: テーマID
        training_job_id: TrainingJob ID
        best_score: 最良スコア
        training_params: 使用したパラメータ（辞書）
    """
    try:
        from data_management.models import Model, ModelTrainData, TrainData, TrainingJob
        from django.utils import timezone
        import json
        
        # TrainingJobを取得
        training_job = TrainingJob.objects.get(id=training_job_id)
        
        # Modelレコード作成
        model = Model.objects.create(
            theme_id=theme_id,
            mlflow_run_id=mlflow_run_id,
            status='completed',
            best_score=best_score,
            training_params=json.dumps(training_params, ensure_ascii=False),
            created_by=training_job.created_by,
        )
        
        # TrainDataの紐付け
        train_data_list = TrainData.objects.filter(
            theme_id=theme_id,
            split__in=['train', 'valid', 'test']
        )
        for train_data in train_data_list:
            ModelTrainData.objects.get_or_create(
                model=model,
                train_data=train_data
            )
        
        # TrainingJob更新
        training_job.model = model
        training_job.status = 'completed'
        training_job.completed_at = timezone.now()
        training_job.save()
        
        logging.getLogger(__name__).info(f"DjangoデータベースにModelレコードを作成しました: {model.id}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Djangoデータベースへの保存エラー: {e}", exc_info=True)


def update_training_job_status(training_job_id: int, status: str, error_message: str = None):
    """
    TrainingJobのステータスを更新
    
    Args:
        training_job_id: TrainingJob ID
        status: ステータス（'running', 'completed', 'failed'等）
        error_message: エラーメッセージ（失敗時）
    """
    try:
        from data_management.models import TrainingJob
        from django.utils import timezone
        
        training_job = TrainingJob.objects.get(id=training_job_id)
        training_job.status = status
        if error_message:
            training_job.error_message = error_message
        if status in ['completed', 'failed', 'cancelled']:
            training_job.completed_at = timezone.now()
        training_job.save()
        
        logging.getLogger(__name__).info(f"TrainingJobステータスを更新しました: {training_job_id} -> {status}")
    except Exception as e:
        logging.getLogger(__name__).error(f"TrainingJobステータス更新エラー: {e}", exc_info=True)


def main():
    """
    メイン関数
    """
    # 引数のパース
    args = parse_args()
    
    # ロギングのセットアップ
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("ハイパーパラメータチューニングスクリプトを開始します")
    logger.info("=" * 80)
    logger.info(f"引数: {vars(args)}")
    
    # Django環境のセットアップ（training_job_idが指定されている場合）
    if args.training_job_id:
        setup_django()
        update_training_job_status(args.training_job_id, 'running')
    
    # params.yamlの上書き（theme_idが指定されている場合）
    params_file = override_params(args.params, args)
    
    try:
        # チューニングの実行
        results = tune(
            params_file=params_file,
            config_file=args.config,
            augments_config=args.augments,
            storage=args.storage,
            study_name=args.study_name,
            load_if_exists=args.load_if_exists,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            checkpoint_dir=args.checkpoint_dir,
            training_job_id=args.training_job_id,
        )
        
        # 結果の表示
        logger.info("=" * 80)
        logger.info("チューニングが完了しました")
        logger.info("=" * 80)
        logger.info(f"親ランID: {results['parent_run_id']}")
        logger.info(f"最良のトライアル: {results['best_trial'].number}")
        logger.info(f"最良のトライアルのMLflow run ID: {results['best_trial_run_id']}")
        logger.info(f"最良の値: {results['best_value']}")
        logger.info(f"最良のパラメータ:")
        for key, value in results['best_params'].items():
            logger.info(f"  {key}: {value}")
        logger.info(f"最良のパラメータファイル: {results['best_params_file']}")
        logger.info("=" * 80)
        
        # Studyの統計情報
        study = results['study']
        logger.info("Study統計情報:")
        logger.info(f"  完了したトライアル数: {len(study.trials)}")
        logger.info(f"  最良の値: {study.best_value}")
        logger.info(f"  最良のトライアル: {study.best_trial.number}")
        logger.info("=" * 80)
        
        # Djangoデータベースに保存（training_job_idが指定されている場合）
        if args.training_job_id and args.theme_id:
            # Optunaで選択された最良のパラメータを使用
            # results['best_params']には、base_paramsにbest_trial.paramsの値を設定したものが含まれている
            best_params = results.get('best_params')
            if best_params is None:
                logger.warning("最良のパラメータが取得できなかったため、params.yamlを使用します")
                import yaml
                with open(params_file, 'r') as f:
                    best_params = yaml.safe_load(f)
            
            child_run_id = results.get('best_trial_run_id') or results.get('parent_run_id')
            if child_run_id is None:
                logger.warning("MLflow Run IDが取得できなかったため、Modelを保存しません")
            else:
                save_model_to_django(
                    mlflow_run_id=child_run_id,
                    theme_id=args.theme_id,
                    training_job_id=args.training_job_id,
                    best_score=results['best_value'],
                    training_params=best_params
                )
        
        # 最良のパラメータで再学習するかの案内
        logger.info("最良のパラメータで再学習する場合:")
        logger.info(f"  cp {results['best_params_file']} params.yaml")
        logger.info(f"  python scripts/train.py")
        
    except Exception as e:
        logger.error(f"チューニング中にエラーが発生しました: {e}", exc_info=True)
        
        # TrainingJobステータスを失敗に更新
        if args.training_job_id:
            update_training_job_status(args.training_job_id, 'failed', str(e))
        
        sys.exit(1)
    
    finally:
        # 一時ファイルの削除
        if params_file != args.params and Path(params_file).exists():
            Path(params_file).unlink()
            logger.info(f"一時ファイルを削除: {params_file}")


if __name__ == "__main__":
    main()
