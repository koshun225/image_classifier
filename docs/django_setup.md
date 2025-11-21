# Django学習データ管理システム - セットアップガイド

最終更新: 2025-11-16

## 📋 概要

このドキュメントでは、Django学習データ管理システムのセットアップ手順と使い方を説明します。

## 🎯 機能

- **テーマ管理**: 分類タスクのテーマを作成・管理
- **ラベル管理**: 各テーマのクラスラベルを作成・管理
- **学習データ管理**: 画像のアップロード、ラベル付け、データ分割
- **モデル管理**: MLflowとの連携、学習済みモデルの管理
- **Django Admin UI**: 非エンジニアでも使いやすい管理画面

## 🚀 セットアップ手順

### 1. 自動セットアップ（推奨）

```bash
# セットアップスクリプトを実行
./scripts/setup_django.sh
```

このスクリプトは以下を実行します：
1. 依存パッケージのインストール
2. マイグレーションファイルの作成
3. マイグレーションの適用
4. 静的ファイルの収集
5. スーパーユーザーの作成（オプション）

### 2. 手動セットアップ

#### 2.1 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

#### 2.2 データベースのマイグレーション

```bash
cd src/web
python manage.py makemigrations
python manage.py migrate
```

#### 2.3 スーパーユーザーの作成

```bash
python manage.py createsuperuser
```

ユーザー名、メールアドレス、パスワードを入力してください。

#### 2.4 静的ファイルの収集

```bash
python manage.py collectstatic --noinput
```

## 🖥️ 管理画面の起動

### 開発サーバーの起動

```bash
cd src/web
python manage.py runserver
```

### アクセス

ブラウザで以下のURLにアクセス：

```
http://127.0.0.1:8000/admin/
```

作成したスーパーユーザーのユーザー名とパスワードでログインしてください。

## 📊 データベース構造

### テーブル一覧

1. **Theme（テーマ）**
   - `id`: 主キー
   - `name`: テーマ名（一意）
   - `description`: 説明
   - `created_at`, `updated_at`: タイムスタンプ

2. **Label（ラベル）**
   - `id`: 主キー
   - `theme_id`: テーマID（外部キー）
   - `label_name`: ラベル名
   - `created_at`: タイムスタンプ

3. **TrainData（学習データ）**
   - `id`: 主キー
   - `image`: 画像ファイル
   - `theme_id`: テーマID（外部キー）
   - `label_id`: ラベルID（外部キー）
   - `split`: データ分割（train/valid/test）
   - `labeled_by`: ラベル付けした人
   - `created_at`, `updated_at`: タイムスタンプ

4. **Model（モデル）**
   - `id`: 主キー
   - `theme_id`: テーマID（外部キー）
   - `mlflow_run_id`: MLflow Run ID
   - `model_name`: モデル名
   - `description`: 説明
   - `created_at`, `updated_at`: タイムスタンプ

5. **ModelTrainData（モデル-学習データ関連）**
   - `model_id`: モデルID（外部キー）
   - `train_data_id`: 学習データID（外部キー）
   - `created_at`: タイムスタンプ

## 🔧 使い方

### 1. テーマの作成

1. 管理画面にログイン
2. 「テーマ」をクリック
3. 「テーマを追加」をクリック
4. テーマ名と説明を入力して保存

例：
- テーマ名: `MNIST`
- 説明: `手書き数字認識`

### 2. ラベルの作成

1. 「ラベル」をクリック
2. 「ラベルを追加」をクリック
3. テーマを選択し、ラベル名を入力して保存

例（MNISTテーマの場合）：
- ラベル名: `0`, `1`, `2`, ..., `9`

### 3. 学習データの登録

1. 「学習データ」をクリック
2. 「学習データを追加」をクリック
3. 以下を入力：
   - テーマ: 選択
   - ラベル: 選択
   - 画像: アップロード
   - ラベル付けした人: 入力（オプション）
4. 保存

**注意**: データ分割（split）は初回学習時に自動的に割り当てられます。

### 4. データ分割の実行

Pythonスクリプトから実行：

```python
import os
import sys
import django

# Django環境のセットアップ
sys.path.insert(0, 'src/web')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from data_management.crud import assign_splits_to_new_data

# テーマID 1 の新規データを分割
train_count, valid_count, test_count = assign_splits_to_new_data(
    theme_id=1,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

print(f"Train: {train_count}, Valid: {valid_count}, Test: {test_count}")
```

## 🤖 PyTorchパイプラインとの統合

### Django DataModuleの使用

```python
from src.data.django_datamodule import DjangoDataModule

# DataModuleを初期化
datamodule = DjangoDataModule(
    theme_id=1,  # テーマID
    batch_size=32,
    augmentation_config='auguments.yaml'
)

# Lightning Trainerで使用
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=10)
trainer.fit(model, datamodule=datamodule)
```

### Django Datasetの使用

```python
from src.data.django_dataset import DjangoClassificationDataset

# Datasetを初期化
train_dataset = DjangoClassificationDataset(
    theme_id=1,
    split='train'
)

# DataLoaderで使用
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

### データセット情報の取得

```python
from src.data.django_dataset import get_dataset_info_from_django

# データセット情報を取得
info = get_dataset_info_from_django(theme_id=1)

print(f"テーマ名: {info['theme_name']}")
print(f"クラス数: {info['num_classes']}")
print(f"クラス名: {info['class_names']}")
print(f"分割統計: {info['split_statistics']}")
```

## 📝 CRUD操作（プログラムから）

### テーマの作成

```python
from data_management.crud import create_theme

theme = create_theme(
    name="動物分類",
    description="犬、猫、鳥の分類"
)
```

### ラベルの作成

```python
from data_management.crud import create_label

label_dog = create_label(theme_id=theme.id, label_name="犬")
label_cat = create_label(theme_id=theme.id, label_name="猫")
label_bird = create_label(theme_id=theme.id, label_name="鳥")
```

### 学習データの作成

```python
from data_management.crud import create_traindata

traindata = create_traindata(
    theme_id=theme.id,
    label_id=label_dog.id,
    image_path="/path/to/dog_image.jpg",
    labeled_by="Tanaka"
)
```

### データ分割統計の取得

```python
from data_management.crud import get_split_statistics

stats = get_split_statistics(theme_id=1)
print(stats)
# {'train': 100, 'valid': 20, 'test': 20, 'unsplit': 10}
```

## 🔄 既存コードからの移行

### 旧方式（ファイルベース）

```python
from src.data.dataset import load_dataset_from_directory

train_data, train_labels = load_dataset_from_directory(
    'data/train',
    class_names=['0', '1', '2']
)
```

### 新方式（Djangoデータベース）

```python
from src.data.django_dataset import DjangoClassificationDataset

train_dataset = DjangoClassificationDataset(
    theme_id=1,
    split='train'
)
```

**メリット**:
- データベースでメタデータを一元管理
- データ分割情報の永続化
- モデルと学習データの関連付け
- 管理画面での視覚的な管理

## 🛠️ トラブルシューティング

### エラー: "No module named 'django'"

```bash
pip install django>=4.2.0
```

### エラー: "DJANGO_SETTINGS_MODULE is not set"

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
```

### エラー: "table data_management_theme doesn't exist"

```bash
cd src/web
python manage.py migrate
```

### 管理画面にアクセスできない

1. サーバーが起動しているか確認
2. URLが正しいか確認（http://127.0.0.1:8000/admin/）
3. スーパーユーザーが作成されているか確認

## 📚 関連ドキュメント

- [要件定義書](requirements.md)
- [アーキテクチャ設計書](architecture.md)
- [進捗管理](progress.md)
- [再現性確保](reproducibility.md)

## 🔗 参考リンク

- [Django公式ドキュメント](https://docs.djangoproject.com/)
- [Django Admin](https://docs.djangoproject.com/en/4.2/ref/contrib/admin/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

