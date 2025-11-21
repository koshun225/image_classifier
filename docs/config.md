# 設定ファイル（config.yaml）

`config.yaml`はプロジェクト全体の設定を管理するファイルです。

## 設定項目

### データディレクトリの設定

```yaml
data:
  # 本番用データディレクトリ（DVC管理用、後方互換性のため残存）
  root_dir: "data"
  
  # DVC管理対象
  dvc_managed:
    - "images"      # Django画像ファイル（テーマごと）
    - "artifacts"   # 前処理済み画像など
```

#### 説明

- **root_dir**: 後方互換性のため残存。現在は使用されていません。
- **dvc_managed**: DVCで管理するディレクトリのリスト。`scripts/manage_dvc.py`で自動追跡に使用されます。
- **画像ファイル**: `images/[theme_id]/`に保存されます（DjangoのFileFieldで管理）。
- **学習データ**: Djangoデータベース（`database.db`）で管理されます。

#### DVC追跡の自動管理

`config.yaml`の`dvc_managed`設定を使って、DVCの追跡を自動化できます：

```bash
# 完全同期（add + commit + push）
python scripts/manage_dvc.py full --push

# または個別に実行
python scripts/manage_dvc.py sync    # config.yamlに基づいてDVC追加
python scripts/manage_dvc.py commit  # .dvcファイルをコミット
python scripts/manage_dvc.py push    # リモートにプッシュ
python scripts/manage_dvc.py list    # 追跡ファイル一覧表示
```

**注意:** `dvc_managed`を編集しても、自動的にDVCに反映されるわけではありません。`manage_dvc.py`スクリプトを実行する必要があります。

**ワークフロー:**
1. `config.yaml`の`dvc_managed`を編集
2. `python scripts/manage_dvc.py full --push` を実行
3. `git add *.dvc .gitignore && git commit -m 'Update DVC tracking'`
4. `git push`

### Djangoベースのデータ管理

本プロジェクトは**Djangoデータベースベース**のデータ管理に完全移行しました。

1. **データベース管理**
   - 学習データ（画像パス、ラベル、分割情報）は`database.db`（SQLite）で管理
   - テーマ、ラベル、データ分割がデータベースで一元管理

2. **画像ファイル管理**
   - 画像ファイルは`images/[theme_id]/`に保存
   - DjangoのFileFieldで自動的にパス管理

3. **テストの独立性**
   - テストはDjangoデータベースを使用
   - ダミーMNIST画像を動的に作成するため、固定ファイル不要

### Artifactsの設定

```yaml
artifacts:
  # 前処理済み画像の保存先（オプション）
  preprocessed_dir: "artifacts/preprocessed"
```

**注意**: 
- データ分割情報はDjangoデータベースで管理されるため、`splits_dir`は廃止されました。
- 前処理済み画像は必要に応じて`artifacts/preprocessed/`に保存できます。

### MLflowの設定

```yaml
mlflow:
  # 実験結果の保存先
  tracking_uri: "experiments/mlruns"
  
  # 実験名
  experiment_name: "classification_with_mlops"
```

### ログ設定

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s [%(levelname)8s] %(message)s"
```

## 使用方法

### Pythonコードから読み込む（Djangoベース）

```python
import yaml

# config.yamlを読み込む
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# MLflow設定を取得
mlflow_tracking_uri = config["mlflow"]["tracking_uri"]
mlflow_experiment_name = config["mlflow"]["experiment_name"]

# データはDjangoデータベースから読み込む
from src.data.dataset import load_dataset, get_dataset_info

# テーマIDを指定してデータセットを作成
dataset = load_dataset(theme_id=1, split='train')
info = get_dataset_info(theme_id=1)

print(f"クラス数: {info['num_classes']}")
print(f"総画像数: {info['total_samples']}")
```

### pytestのfixtureから使用（Djangoベース）

`tests/conftest.py`で定義されているfixtureを使用します：

```python
import pytest

def test_something(test_theme, mnist_test_data):
    # test_theme: Djangoテストテーマとラベル
    # mnist_test_data: ダミーMNIST画像データ（Djangoデータベース登録済み）
    
    theme, labels = test_theme
    print(f"テストテーマ: {theme.name}")
    print(f"ラベル数: {len(labels)}")
```

## カスタマイズ

プロジェクトの要件に応じて、`config.yaml`を編集してください：

### DVCでの画像管理ディレクトリを変更

```yaml
data:
  dvc_managed:
    - "images"      # Django画像ファイル
    - "artifacts"   # 前処理済み画像など
    - "your_custom_dir"  # カスタムディレクトリを追加
```

### MLflowの設定を変更

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"  # MLflow serverのURL
  experiment_name: "my_experiment"
```

## 注意事項

- `config.yaml`はGitで管理されます
- パスは相対パスまたは絶対パスで指定できます
- 環境ごとに異なる設定が必要な場合は、環境変数や別の設定ファイルを使用することを検討してください

## 関連ファイル

- `params.yaml`: ハイパーパラメータ設定（モデル学習に関する設定）
- `.dvcignore`: DVC管理から除外するファイル
- `.gitignore`: Git管理から除外するファイル

