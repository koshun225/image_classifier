#!/usr/bin/env python3
"""
学習実行スクリプト

コマンドラインから学習を実行します。

使用例:
    # デフォルト設定で実行（params.yamlのtheme_idを使用）
    python scripts/train.py
    
    # 特定のテーマIDを指定して実行
    python scripts/train.py --theme-id 7
    
    # パラメータをカスタマイズして実行
    python scripts/train.py --theme-id 7 --epochs 50 --batch-size 64
    
    # MLflow run名を指定して実行
    python scripts/train.py --theme-id 7 --run-name "experiment_001"
    
    # MLflowなしで実行
    python scripts/train.py --theme-id 7 --no-mlflow
    
    # カスタム設定ファイルを使用
    python scripts/train.py --params custom_params.yaml --config custom_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import train


def parse_args():
    """
    コマンドライン引数をパース
    
    Returns:
        argparse.Namespace: パースされた引数
    """
    parser = argparse.ArgumentParser(
        description="画像分類モデルの学習",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 設定ファイル
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="params.yamlファイルのパス"
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
        help="Djangoテーマ ID（params.yamlを上書き）"
    )
    
    # 学習設定
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="エポック数（params.yamlを上書き）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="バッチサイズ（params.yamlを上書き）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学習率（params.yamlを上書き）"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoaderのワーカー数"
    )
    
    # ディレクトリ設定
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="チェックポイントの保存ディレクトリ"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="ログの保存ディレクトリ"
    )
    
    # MLflow設定
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="MLflowを無効化"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run名"
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="決定論的な学習（再現性）"
    )
    
    # その他
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="モニターするメトリクス"
    )
    parser.add_argument(
        "--use-preprocessing",
        action="store_true",
        help="前処理を使用する"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベル"
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
    """
    import yaml
    
    # params.yamlを読み込み
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    
    # コマンドライン引数で上書き
    modified = False
    
    if args.theme_id is not None:
        params.setdefault("data", {})["theme_id"] = args.theme_id
        modified = True
        logging.info(f"テーマ IDを上書き: {args.theme_id}")
    
    if args.epochs is not None:
        params.setdefault("training", {})["num_epochs"] = args.epochs
        modified = True
        logging.info(f"エポック数を上書き: {args.epochs}")
    
    if args.batch_size is not None:
        params.setdefault("training", {})["batch_size"] = args.batch_size
        modified = True
        logging.info(f"バッチサイズを上書き: {args.batch_size}")
    
    if args.learning_rate is not None:
        params.setdefault("training", {})["learning_rate"] = args.learning_rate
        modified = True
        logging.info(f"学習率を上書き: {args.learning_rate}")
    
    if args.num_workers is not None:
        params.setdefault("training", {})["num_workers"] = args.num_workers
        modified = True
        logging.info(f"ワーカー数を上書き: {args.num_workers}")
    
    if args.run_name is not None:
        params.setdefault("training", {})["run_name"] = args.run_name
        modified = True
        logging.info(f"MLflow run名を上書き: {args.run_name}")
    
    # 上書きされた場合、一時ファイルに保存
    if modified:
        temp_params_file = f"{params_file}.tmp"
        with open(temp_params_file, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        logging.info(f"上書きされたパラメータを保存: {temp_params_file}")
        return temp_params_file
    
    return params_file


def main():
    """
    メイン関数
    """
    # 引数のパース
    args = parse_args()
    
    # ロギングのセットアップ
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("学習スクリプトを開始します")
    logger.info(f"引数: {vars(args)}")
    
    # パラメータの上書き
    params_file = override_params(args.params, args)
    
    try:
        # 学習の実行
        results = train(
            params_file=params_file,
            config_file=args.config,
            augments_config=args.augments,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            enable_mlflow=not args.no_mlflow,
            run_name=args.run_name,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            deterministic=args.deterministic,
            monitor=args.monitor,
            use_preprocessing=args.use_preprocessing,
        )
        
        # 結果の表示
        logger.info("=" * 80)
        logger.info("学習が完了しました")
        logger.info("=" * 80)
        logger.info(f"テスト結果: {results['test_results']}")
        logger.info(f"最良モデルパス: {results['best_model_path']}")
        logger.info(f"MLflow run ID: {results['mlflow_run_id']}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # 一時ファイルの削除
        if params_file != args.params and Path(params_file).exists():
            Path(params_file).unlink()
            logger.info(f"一時ファイルを削除: {params_file}")


if __name__ == "__main__":
    main()
