# Image Classifier with MLOps

深層学習ベースの画像分類パイプラインを、データ／モデル／コードの完全なバージョン管理とともに提供するMLOpsテンプレートです。Djangoでのデータ収集からPyTorch Lightningでの学習、Optunaによる自動チューニング、MLflowとDVCによるトラッキングまでを一貫して扱えます。

## 主な特徴

- Django製ラベリングUI（Label Studio風）によるテーマ・ラベル管理とデータ分割
- PyTorch Lightningベースの学習・推論モジュールとOptunaチューナー
- MLflowでの実験ログ、DVCでの大規模データ管理、Gitでのコード管理
- 可視化スクリプトによる前処理／オーグメンテーション確認
- pytest＋Djangoテストデータベースによる包括的な自動テスト

## 技術スタック

| カテゴリ | 使用技術 |
| --- | --- |
| 深層学習 | PyTorch Lightning |
| Web/データ管理 | Django, Django ORM (SQLite/PostgreSQL) |
| 実験管理 | MLflow |
| データバージョン管理 | DVC |
| ハイパーパラメータ探索 | Optuna |
| インフラ | Git, シェルスクリプト群 |

## ディレクトリ構造

```
image_classifier/
├── src/                # データ処理・モデル・学習・Webアプリ
│   ├── data/           # Dataset/DataModule/前処理
│   ├── models/         # ResNet等のモデル定義
│   ├── training/       # LightningModuleと学習エントリポイント
│   ├── tuning/         # Optuna連携
│   ├── utils/          # 共通ユーティリティ
│   └── web/            # Djangoアプリケーション
├── scripts/            # セットアップ／学習／可視化などのCLI
├── docs/               # 各種ガイド・設計ドキュメント
├── images/, artifacts/ # 元データ・前処理成果物
├── experiments/mlruns/ # MLflowローカルストア
├── tests/              # pytest＋Django DBベースのテスト群
├── config.yaml         # データ／実験設定
├── params.yaml         # 学習パラメータ
├── auguments.yaml      # 前処理・Aug設定
└── requirements.txt    # 依存パッケージ
```

## セットアップ手順

```bash
# 1. 依存関係のインストール（自前のPython環境を用意）
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Django + DB 初期化
./scripts/setup_django.sh

# 3. （任意）PyTorchを別途インストールする場合
./scripts/install_pytorch.sh
```

## データ管理とラベリング

### Label-Studio風ラベリングUI（推奨）

```bash
cd src/web
python manage.py runserver
# http://127.0.0.1:8000/ にアクセス
```

- テーマ作成 → ラベル定義 → 画像アップロード → ラベリングをWeb UIで完結
- ドラッグ＆ドロップアップロード、ショートカット、リアルタイム統計を備えたダッシュボード
- テーマ詳細からボタン一つでTrain/Val/Testの自動分割
- 詳細は `docs/labeling_ui_guide.md` と `docs/django_setup.md` を参照

### Django Admin

```bash
# http://127.0.0.1:8000/admin/ にアクセスし、スーパーユーザーでログイン
```

管理画面経由でも同じデータが扱えます。

## 学習・推論ワークフロー

### 1. データ分割と準備

```bash
# Web UIから分割するか、スクリプトで明示的に実行
python -c "from src.data.split import split_dataset; split_dataset(theme_id=1)"
```

### 2. モデル学習

```bash
# ベーシックな学習
python scripts/train.py --theme-id 1 --epochs 100

# 代表的なオプション
python scripts/train.py \
  --theme-id 1 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --accelerator gpu \
  --devices 0 \
  --precision 16-mixed

# ハイパーパラメータチューニング
python scripts/tune.py --theme-id 1 --n-trials 50
cp params_best.yaml params.yaml
python scripts/train.py --theme-id 1
```

`scripts/train.py --help` と `scripts/tune.py --help` で全オプションを確認できます。

## 実験管理（MLflow）

| 用途 | コマンド |
| --- | --- |
| ローカルUIで簡単に確認 | `mlflow ui --backend-store-uri experiments/mlruns` |
| 推奨の専用サーバー | `./scripts/start_mlflow.sh [--background]` |

学習コマンドを実行すると自動でMLflowへログされます。`http://localhost:5000` から精度・損失・ハイパーパラメータ・チェックポイントを参照してください。詳細手順は `docs/mlflow_setup.md`。

## データ／モデルのバージョン管理（DVC）

```bash
dvc init
dvc remote add -d myremote s3://mybucket/dvcstore
python scripts/manage_dvc.py sync     # 変更の追加
python scripts/manage_dvc.py commit   # メタデータ更新
python scripts/manage_dvc.py push     # リモートへアップロード
python scripts/manage_dvc.py full --push  # 一括処理
```

Gitには `.dvc` メタファイルと設定類のみコミットし、`database.db` などは除外します。詳細は `scripts/README.md` 参照。

## 可視化とデバッグ

`auguments.yaml` の挙動を画像で確認できるツールを `scripts/visualization/` に用意しています。すべて `--theme-id` でDjango DB上のテーマを指定します。

```bash
python scripts/visualization/vis_preprocessing.py --theme-id 1
python scripts/visualization/vis_augmentation.py --theme-id 1
python scripts/visualization/vis_dataset.py --theme-id 1
```

生成された画像は `workspace/` に保存されます。詳細は `scripts/visualization/README.md`。

## テスト

```bash
./scripts/setup_django.sh  # テストDB初期化（初回のみ）
pytest                     # すべてのテスト
pytest -v                  # 詳細ログ
pytest tests/test_pytorch_dataset.py -v           # 個別テスト
pytest --cov=src --cov-report=html                # カバレッジ
```

テストはDjango DBとダミーMNIST画像を動的に利用します。挙動・前提条件は `tests/README.md`・`tests/RUN_TESTS.md` に記載しています。

## 参考ドキュメント

- `docs/architecture.md`：全体アーキテクチャ
- `docs/requirements.md`：要件とユースケース
- `docs/config.md`：`config.yaml` の詳細
- `docs/data_pipeline.md`：データパイプライン手順
- `docs/reproducibility.md`：再現性ガイド
- `docs/progress.md`：開発進捗ログ
- `docs/mlflow_setup.md` / `docs/django_setup.md`：それぞれのセットアップ手順

このREADMEでは概要を扱い、詳細なノウハウは `docs/` フォルダに集約しています。必要に応じて参照しながらワークフローを構築してください。

