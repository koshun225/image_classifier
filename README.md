# Classification with MLOps

深層学習を用いた画像分類モデルの学習・推論システム

## 概要

本プロジェクトは、MLOpsのベストプラクティスに基づき、コード、データ、モデルのバージョン管理を行い、再現性の高い機械学習パイプラインを実現します。

## 技術スタック

- **深層学習フレームワーク**: PyTorch Lightning
- **Webフレームワーク**: Django (学習データ管理)
- **バージョン管理**:
  - Git: コードのバージョン管理
  - DVC: データのバージョン管理
  - MLflow: モデルのバージョン管理と実験管理
- **データ管理**: Django ORM (SQLite/PostgreSQL)
- **ハイパーパラメータチューニング**: Optuna

## プロジェクト構造

```
classification_with_mlops/
├── src/              # ソースコード
│   ├── data/        # データ処理モジュール（Djangoベース）
│   ├── models/      # モデル定義
│   ├── training/    # 学習モジュール
│   ├── tuning/      # ハイパーパラメータチューニング
│   ├── utils/       # ユーティリティ
│   └── web/         # Django Webアプリケーション
├── images/           # 画像ファイル保存ディレクトリ（テーマごと）
├── artifacts/        # 成果物（前処理済み画像など）
├── experiments/      # MLflow実験結果
├── tests/            # テストコード（Djangoベース）
├── workspace/        # データ準備・デモスクリプト
├── scripts/          # 実行スクリプト
├── database.db       # Django SQLiteデータベース
├── config.yaml       # プロジェクト設定
├── params.yaml       # ハイパーパラメータ設定
└── auguments.yaml    # データオーグメンテーション設定
```

詳細は以下のドキュメントを参照してください：
- `docs/requirements.md`: 要件定義
- `docs/architecture.md`: アーキテクチャ設計
- `docs/config.md`: 設定ファイル（config.yaml）の説明
- `docs/reproducibility.md`: 再現性の確保
- `docs/progress.md`: **実装進捗管理** ← 進捗確認はこちら
- `docs/data_pipeline.md`: データパイプライン使い方ガイド
- `docs/django_setup.md`: **Django学習データ管理システムのセットアップ** ← NEW！
- `docs/mlflow_setup.md`: MLflowセットアップガイド

## クイックスタート

### 🆕 Django学習データ管理システム（推奨）

**2つの管理方法を提供：**

#### 1. Label-Studio風ラベリングUI ⭐️ NEW!

直感的なWebUIで効率的にラベリング！

```bash
# 1. Django環境のセットアップ
./scripts/setup_django.sh

# 2. Webサーバーを起動
cd src/web
python manage.py runserver

# 3. ブラウザでアクセス
http://127.0.0.1:8000/

# 4. ダッシュボードからラベリング開始！
```

**主な機能：**
- 🎯 **Label-Studio風ワークフロー**: テーマ作成 → ラベル追加 → 画像アップロード → ラベリング
- 📤 **ドラッグ&ドロップアップロード**: 複数画像を一括アップロード
- 📸 画像グリッド表示
- 🏷️ ワンクリックラベリング
- ⌨️ キーボードショートカット（1-9でラベル選択、Spaceで次へ）
- 🔀 自動データ分割
- 📊 リアルタイム統計表示
- ⚙️ **Admin完全不要**: すべての操作がWebUIで完結

詳細は `docs/labeling_ui_guide.md` を参照してください。

#### 2. Django Admin管理画面

従来の管理画面でも操作可能：

```bash
# http://127.0.0.1:8000/admin/
# スーパーユーザーでログイン
```

詳細は `docs/django_setup.md` を参照してください。

---

### 本番用（実際のデータで学習）

```bash
# 1. Django環境のセットアップ
./scripts/setup_django.sh

# 2. WebUIでテーマ作成・ラベル追加・画像アップロード
cd src/web
python manage.py runserver
# http://127.0.0.1:8000/ でラベリング

# 3. データ分割（WebUIまたはスクリプトで実行）
# WebUIの場合: テーマ詳細画面でデータ分割ボタンをクリック
# スクリプトの場合:
# python -c "from src.data.split import split_dataset; split_dataset(theme_id=1)"

# 4. 学習を実行
python scripts/train.py --theme-id 1

# 5. DVCで画像ファイルを管理（オプション）
dvc init  # 初回のみ
dvc remote add -d myremote s3://mybucket/dvcstore  # 初回のみ
python scripts/manage_dvc.py full --push

# 6. Gitにコミット（データベースは除外）
git add config.yaml params.yaml *.dvc .gitignore
git commit -m 'Update configuration and DVC tracking'
git push
```

### テスト用（動作確認）

```bash
# 1. Django環境のセットアップ
./scripts/setup_django.sh

# 2. テストの実行（pytest - Djangoベース）
pytest

# または特定のテストファイルのみ
pytest tests/test_split_consistency.py -v
pytest tests/test_real_dataset.py -v
pytest tests/test_pytorch_dataset.py -v

# テストはDjangoデータベースを使用し、ダミーMNIST画像を動的に作成します
```

### データパイプライン動作確認（可視化スクリプト）

`auguments.yaml` の設定を視覚的に確認できる可視化スクリプトを用意しています（**Djangoベース**）。

```bash
# 前処理の動作確認（auguments.yamlのpreprocessing設定を使用）
python scripts/visualization/vis_preprocessing.py --theme-id 1

# オーグメンテーションの動作確認（auguments.yamlのtrain/val/test設定を使用）
python scripts/visualization/vis_augmentation.py --theme-id 1

# Dataset/DataModuleの動作確認（auguments.yamlの全設定を使用）
python scripts/visualization/vis_dataset.py --theme-id 1
```

**重要**:
- 可視化スクリプトはDjangoデータベースからテーマを指定してデータを取得します
- `--theme-id`引数でテーマIDを指定してください（WebUIで確認可能）
- `auguments.yaml` でパラメータを変更すると、可視化の挙動が変わります
- 生成された画像は `workspace/` ディレクトリに保存されます

詳細は `scripts/visualization/README.md` を参照してください。

## MLflowのセットアップ

MLflowは実験管理ツールです。学習のメトリクス、ハイパーパラメータ、モデルを記録・管理できます。

### 方法1: サーバーなしで使用（簡単）

```bash
# そのまま学習を実行（サーバー不要）
python scripts/train.py --epochs 10

# 学習後にUIで結果を確認
mlflow ui --backend-store-uri experiments/mlruns
# ブラウザで http://localhost:5000 を開く
```

### 方法2: MLflow Serverを使用（推奨）

```bash
# 1. MLflow Serverを起動
./scripts/start_mlflow.sh

# または、バックグラウンドで起動
./scripts/start_mlflow.sh --background

# 2. ブラウザでUIを開く
# http://localhost:5000

# 3. 別のターミナルで学習を実行
python scripts/train.py --epochs 10

# 4. リアルタイムで結果を確認
```

**詳細は `docs/mlflow_setup.md` を参照してください。**

## 学習の実行

### 基本的な学習

```bash
# 1. MLflow Serverを起動（推奨）
./scripts/start_mlflow.sh

# 2. 学習を実行
python scripts/train.py --epochs 100

# 3. ブラウザでリアルタイムに結果を確認
# http://localhost:5000
```

### パラメータを変更して学習

```bash
# エポック数、バッチサイズ、学習率を指定
python scripts/train.py --epochs 50 --batch-size 64 --learning-rate 0.001

# GPUを指定
python scripts/train.py --accelerator gpu --devices 0

# 混合精度で高速化
python scripts/train.py --precision 16-mixed
```

### ハイパーパラメータチューニング

```bash
# Optunaで自動チューニング
python scripts/tune.py --n-trials 50

# 最良のパラメータが params_best.yaml に保存されます
cp params_best.yaml params.yaml
python scripts/train.py
```

詳細なオプションは以下を参照：
```bash
python scripts/train.py --help
python scripts/tune.py --help
```

### 完全なワークフロー

```bash
# 1. データを配置
# data/ver1/, data/ver2/ などにデータを配置

# 2. データを分割（自動的に config.yaml に記録）
python scripts/split_data.py

# 3. DVCで管理（初回のみ初期化とリモート設定）
dvc init  # 初回のみ
dvc remote add -d myremote s3://mybucket/dvcstore  # 初回のみ
python scripts/manage_dvc.py full --push

# 4. Gitにコミット（split_config.yaml も含める）
git add config.yaml artifacts/splits/*.txt artifacts/splits/split_config.yaml *.dvc .gitignore
git commit -m 'Add data and splits (ver1, ver2, ver3)'
git push

# 5. 学習実行（今後実装）
# python scripts/train.py
```

### DVCコマンド詳細

```bash
python scripts/manage_dvc.py sync    # DVC追加
python scripts/manage_dvc.py commit  # DVCコミット
python scripts/manage_dvc.py push    # DVCプッシュ
python scripts/manage_dvc.py list    # 追跡ファイル一覧
python scripts/manage_dvc.py full --push  # 完全同期
```

詳細は `scripts/README.md` を参照してください。

## テスト

```bash
# すべてのテストを実行
pytest

# 詳細な出力で実行
pytest -v

# カバレッジレポート付きで実行
pytest --cov=src --cov-report=html

# 特定のテストのみ実行
pytest tests/test_split_consistency.py::test_train_split_consistency
```

詳細は `tests/README.md` を参照してください。

