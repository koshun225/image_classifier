# Scripts - 実行スクリプト

このディレクトリには、プロジェクトで使用する実行スクリプトが含まれています。

## スクリプト一覧

### train.py

学習を実行するスクリプトです（**Djangoベース**）。

**機能:**
- Djangoデータベースからテーマを指定して学習データを取得
- PyTorch Lightningによる学習実行
- MLflowへのメトリクス・モデル記録
- Git commit IDによる再現性の保証

**使用方法:**

```bash
# 基本的な使用（params.yamlの設定を使用、テーマID指定）
python scripts/train.py --theme-id 1

# エポック数を指定
python scripts/train.py --theme-id 1 --epochs 50

# MLflow Run名を指定
python scripts/train.py --theme-id 1 --run-name "baseline_experiment"

# params.yamlパスを指定
python scripts/train.py --theme-id 1 --params-file custom_params.yaml
```

**前提条件:**
1. Django環境のセットアップ完了（`./scripts/setup_django.sh`）
2. WebUIまたはスクリプトでデータ分割実行済み
3. テーマIDの確認（WebUIまたは`python scripts/check_theme_data.py`）

詳細は `docs/training_guide.md` を参照してください。

---

### tune.py

Optunaを使用したハイパーパラメータチューニングスクリプトです。

**機能:**
- `optuna_params.yaml`からチューニング設定を読み込み
- 各トライアルで`params.yaml`を動的に生成
- 最良のパラメータとモデルをMLflowに記録

**使用方法:**

```bash
# 基本的な使用（optuna_params.yamlの設定を使用）
python scripts/tune.py --theme-id 1

# トライアル数を指定
python scripts/tune.py --theme-id 1 --n-trials 100

# Optunaストレージを指定
python scripts/tune.py --theme-id 1 --storage sqlite:///optuna.db
```

---

### setup_django.sh

Django環境を自動セットアップするスクリプトです。

**機能:**
- データベースマイグレーション実行
- スーパーユーザー作成
- 静的ファイルの収集

**使用方法:**

```bash
# 初回セットアップ
./scripts/setup_django.sh

# スーパーユーザー作成をスキップ
./scripts/setup_django.sh --skip-superuser
```

---

### 可視化スクリプト（visualization/）

データパイプラインの動作確認用スクリプトです（**Djangoベース**）。

**使用方法:**

```bash
# 前処理の可視化
python scripts/visualization/vis_preprocessing.py --theme-id 1

# オーグメンテーションの可視化
python scripts/visualization/vis_augmentation.py --theme-id 1

# Dataset/DataModuleの動作確認
python scripts/visualization/vis_dataset.py --theme-id 1
```

詳細は `scripts/visualization/README.md` を参照してください。

### manage_dvc.py

DVCの追跡管理を自動化するスクリプトです。

**機能:**
- `config.yaml`の`dvc_managed`設定に基づいてDVCに自動追加
- DVC追跡ファイルの一覧表示
- .dvcファイルの自動Gitコミット準備

**使用方法:**

```bash
# config.yamlの設定に基づいてDVCに追加
python scripts/manage_dvc.py sync

# DVC追跡ファイルの一覧表示
python scripts/manage_dvc.py list

# .dvcファイルをコミット
python scripts/manage_dvc.py commit

# リモートストレージにプッシュ
python scripts/manage_dvc.py push

# 完全同期（add + commit + push）
python scripts/manage_dvc.py full --push

# カスタム設定ファイルを使用
python scripts/manage_dvc.py sync --config my_config.yaml

# 特定のリモートにプッシュ
python scripts/manage_dvc.py push --remote myremote
```

**実行例:**

```bash
$ python scripts/manage_dvc.py sync

============================================================
DVC追跡管理スクリプト
============================================================

config.yamlを読み込み中...
DVC管理対象: data, artifacts

============================================================
DVC追跡の同期
============================================================

data をDVCに追加中...
実行中: dvc add data
✓ data をDVCに追加しました。
✓ data.dvc をGitに追加しました。

artifacts をDVCに追加中...
実行中: dvc add artifacts
✓ artifacts をDVCに追加しました。
✓ artifacts.dvc をGitに追加しました。

============================================================
結果
============================================================
成功: 2個
スキップ: 0個
失敗: 0個

✓ すべてのパスがDVCで追跡されています。

============================================================
次のステップ
============================================================
以下のコマンドで.dvcファイルをコミットしてください:
  git add *.dvc .gitignore
  git commit -m 'Add data tracking with DVC'
```

### コマンド一覧

| コマンド | 説明 | DVCコマンド |
|---------|------|------------|
| `sync` | config.yamlの設定に基づいてDVCに追加 | `dvc add` |
| `list` | DVC追跡ファイルの一覧表示 | - |
| `commit` | .dvcファイルをコミット | `dvc commit` |
| `push` | リモートストレージにプッシュ | `dvc push` |
| `full` | 完全同期（add + commit + push） | `dvc add` + `dvc commit` + `dvc push` |

### 前提条件

- DVCが初期化されていること（`dvc init`を実行済み）
- `config.yaml`に`dvc_managed`設定があること
- プッシュする場合は、DVCリモートが設定されていること

### config.yamlの設定例

```yaml
data:
  dvc_managed:
    - "data"
    - "artifacts"
```

### ワークフロー

#### 初回セットアップ

```bash
# 1. DVCを初期化
dvc init
git add .dvc
git commit -m 'Initialize DVC'

# 2. リモートストレージを設定
dvc remote add -d myremote s3://mybucket/dvcstore
# または
dvc remote add -d myremote /path/to/remote/storage
git add .dvc/config
git commit -m 'Configure DVC remote'

# 3. データをDVCに追加
python scripts/manage_dvc.py full --push

# 4. Gitにコミット
git add *.dvc .gitignore
git commit -m 'Add data tracking with DVC'
git push
```

#### 日常的な使用

```bash
# データを更新した場合
python scripts/manage_dvc.py full --push

# Gitにコミット
git add *.dvc .gitignore
git commit -m 'Update data'
git push
```

#### 段階的な実行

```bash
# 1. DVCに追加
python scripts/manage_dvc.py sync

# 2. コミット
python scripts/manage_dvc.py commit

# 3. プッシュ
python scripts/manage_dvc.py push

# 4. Gitにコミット
git add *.dvc .gitignore
git commit -m 'Update DVC tracking'
git push
```

### 追跡対象の追加・削除

1. `config.yaml`の`dvc_managed`リストを編集
2. `python scripts/manage_dvc.py full --push`を実行
3. 変更をGitにコミット

**例: artifactsを追跡から除外する場合**

```yaml
data:
  dvc_managed:
    - "data"
    # - "artifacts"  # コメントアウトまたは削除
```

その後、手動でDVCから削除：
```bash
dvc remove artifacts.dvc
git rm artifacts.dvc
git commit -m 'Remove artifacts from DVC'
```

### トラブルシューティング

#### DVCが初期化されていない

```
エラー: DVCが初期化されていません。
```

**解決方法:**
```bash
dvc init
git add .dvc/config .dvc/.gitignore
git commit -m 'Initialize DVC'
```

#### ディレクトリが存在しない

```
警告: data は存在しません。スキップします。
```

**解決方法:**
- ディレクトリを作成するか、config.yamlから削除してください

#### 既に追跡されている

```
情報: data は既にDVCで追跡されています。
```

**解決方法:**
- これは正常です。既に追跡されている場合はスキップされます。

#### DVCリモートが設定されていない

```
警告: DVCリモートが設定されていません。
```

**解決方法:**
```bash
# ローカルストレージの場合
dvc remote add -d myremote /path/to/remote/storage

# S3の場合
dvc remote add -d myremote s3://mybucket/dvcstore

# 設定をGitにコミット
git add .dvc/config
git commit -m 'Configure DVC remote'
```

## ワークフロー例（Djangoベース）

### セットアップから学習まで

```bash
# 1. Django環境のセットアップ
./scripts/setup_django.sh

# 2. WebUIでテーマ作成・ラベル追加・画像アップロード
cd src/web
python manage.py runserver
# http://127.0.0.1:8000/ でラベリング

# 3. データ分割（WebUIまたはスクリプトで実行）
# WebUIの場合: テーマ詳細画面でデータ分割ボタンをクリック

# 4. 学習を実行
python scripts/train.py --theme-id 1

# 5. DVCで画像ファイルを管理（オプション）
python scripts/manage_dvc.py full --push

# 6. Gitにコミット
git add config.yaml params.yaml *.dvc .gitignore
git commit -m 'Update configuration and DVC tracking'
git push
```

### ハイパーパラメータチューニング

```bash
# 1. optuna_params.yamlを編集（チューニング対象パラメータを設定）

# 2. チューニング実行
python scripts/tune.py --theme-id 1 --n-trials 50

# 3. MLflow UIで最良パラメータを確認
mlflow ui --backend-store-uri experiments/mlruns
```

### 新しいデータを追加

```bash
# 1. WebUIで画像を追加アップロード
# http://127.0.0.1:8000/theme/<theme_id>/

# 2. データ分割を再実行（新しい画像のみ分割）
# WebUIでデータ分割ボタンをクリック

# 3. 再学習を実行
python scripts/train.py --theme-id 1

# 4. DVCを更新（画像ファイルが追加された場合）
python scripts/manage_dvc.py full --push
```

## 今後追加予定のスクリプト

- `register_model.py`: MLflow Model Registryへのモデル登録スクリプト（将来実装）

## 参考ドキュメント

- `docs/training_guide.md`: 学習の詳細ガイド
- `docs/mlflow_setup.md`: MLflowセットアップガイド
- `docs/django_setup.md`: Django環境セットアップガイド
- `scripts/visualization/README.md`: 可視化スクリプトの詳細

