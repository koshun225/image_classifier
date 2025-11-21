# データパイプライン完成ガイド

> **⚠️ 重要な注意事項**
> 
> 本ドキュメントは**旧ファイルベース実装**（v1/v2/v3フォルダ、artifacts/splits/、ClassificationDataset.from_split_file()）のデータパイプラインについて説明しています。
> 
> **現在のプロジェクトは Djangoデータベースベースに完全移行しました。**
> 
> **Djangoベースのデータパイプライン：**
> - データソース：Djangoデータベース（`database.db`）
> - データセット作成：`ClassificationDataset(theme_id=1, split='train')`
> - DataModule作成：`ClassificationDataModule(theme_id=1)`
> - 可視化スクリプト：`scripts/visualization/vis_*.py --theme-id 1`
> 
> 詳細は以下を参照してください：
> - `docs/django_setup.md`: Django環境のセットアップ
> - `scripts/visualization/README.md`: 可視化スクリプトの使い方
> - `src/data/dataset.py`: Djangoベースのデータセット実装
> 
> 以下は参考資料として残しています。

---

データレイヤーの実装が完了しました！このドキュメントでは、実装したデータパイプラインの使い方を説明します（旧ファイルベース実装）。

## 📦 実装したコンポーネント

### 1. 設定ファイル

#### `params.yaml`
ハイパーパラメータの設定ファイル

- モデル設定（name, num_classes）
- 学習設定（batch_size, learning_rate, num_epochs, optimizer）
- データ分割設定（split_ratio, seed）

#### `auguments.yaml`
データオーグメンテーションと前処理の設定ファイル

- 学習時/検証時/テスト時で異なる変換を定義
- albumentationsとtorchvisionの両方に対応
- **preprocessingセクション**: 前処理のパラメータを定義
  - `histogram_equalization`: ヒストグラム均等化の設定（method, clip_limit, tile_grid_size）
  - `gamma_correction`: ガンマ補正の設定（gamma値）
  - `patching`: パッチ化の設定（patch_size, stride, padding）

### 2. データ処理モジュール

#### `src/data/preprocessing.py`
画像の前処理機能

- `HistogramEqualization`: ヒストグラム均等化（CLAHE対応）
- `GammaCorrection`: ガンマ補正
- `Patching`: 画像のパッチ化
- `PreprocessingPipeline`: 前処理の組み合わせ

#### `src/data/augmentation.py`
データオーグメンテーション

- `AugmentationBuilder`: auguments.yamlから変換を構築
- `get_transforms()`: 学習/検証/テスト用の変換を取得
- albumentationsとtorchvisionの自動切り替え

#### `src/data/dataset.py`
PyTorch Dataset

- `ClassificationDataset`: 画像分類用Dataset
- 分割ファイル（train_list.txt）からの読み込み
- 前処理とオーグメンテーションの統合
- albumentationsとtorchvision両方に対応

#### `src/data/datamodule.py`
PyTorch Lightning DataModule

- `ClassificationDataModule`: Lightning DataModule
- データの読み込み、前処理、DataLoaderの作成を一元管理
- `setup()`, `train_dataloader()`, `val_dataloader()`, `test_dataloader()`

## 🚀 使い方

### 基本的な使用例

```python
from src.data.datamodule import ClassificationDataModule

# DataModuleを作成
dm = ClassificationDataModule(
    splits_dir="artifacts/splits",
    augments_config="auguments.yaml",
    batch_size=32,
    num_workers=4,
    use_preprocessing=False  # 前処理を使う場合はTrue
)

# セットアップ
dm.setup("fit")

# DataLoaderを取得
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

# クラス情報を取得
num_classes = dm.get_num_classes()
class_names = dm.get_class_names()

print(f"クラス数: {num_classes}")
print(f"クラス名: {class_names}")

# データを反復
for batch_idx, (images, labels) in enumerate(train_loader):
    # images: (batch_size, 3, 224, 224)
    # labels: (batch_size,)
    print(f"Batch {batch_idx}: {images.shape}, {labels}")
    break
```

### PyTorch Lightning Trainerと統合

```python
import pytorch_lightning as pl
from src.data.datamodule import ClassificationDataModule

# DataModule
dm = ClassificationDataModule(
    splits_dir="artifacts/splits",
    batch_size=32,
    num_workers=4
)

# モデル（後で実装）
# model = ClassificationModel(num_classes=dm.get_num_classes())

# Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices=1
)

# 学習
# trainer.fit(model, dm)

# テスト
# trainer.test(model, dm)
```

### 前処理を使う場合

**重要**: 前処理のパラメータは`auguments.yaml`の`preprocessing`セクションで設定します。

```yaml
# auguments.yaml
preprocessing:
  histogram_equalization:
    enabled: true
    method: "clahe"  # "global" or "clahe"
    clahe:
      clip_limit: 2.0
      tile_grid_size: [8, 8]
  
  gamma_correction:
    enabled: true
    gamma: 1.2  # < 1.0: 明るく、> 1.0: 暗く
  
  patching:
    enabled: false
    patch_size: [14, 14]  # [height, width]
    stride: [7, 7]  # Noneの場合はpatch_sizeと同じ
    padding: true
```

```python
# DataModule作成時に use_preprocessing=True
dm = ClassificationDataModule(
    splits_dir="artifacts/splits",
    use_preprocessing=True  # ← 前処理を有効化
)

# セットアップ時にauguments.yamlから前処理設定を読み込みます
dm.setup("fit")
```

### カスタムデータセットを使う場合

```python
from src.data.dataset import ClassificationDataset
from src.data.augmentation import get_transforms

# 変換を取得
train_transform = get_transforms("auguments.yaml", split="train")

# Datasetを作成（分割ファイルから）
dataset = ClassificationDataset.from_split_file(
    split_file="artifacts/splits/train_list.txt",
    transform=train_transform
)

# または、辞書から作成
data_dict = {
    "0": ["path/to/image1.jpg", "path/to/image2.jpg"],
    "1": ["path/to/image3.jpg", "path/to/image4.jpg"],
}

dataset = ClassificationDataset(
    data_dict=data_dict,
    transform=train_transform
)

# DataLoaderを作成
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## ⚙️ 設定のカスタマイズ

### auguments.yamlで前処理パラメータを調整

前処理のパラメータ（パッチサイズ、ガンマ値、CLAHE設定など）は`auguments.yaml`の`preprocessing`セクションで調整できます。

```yaml
# auguments.yaml
preprocessing:
  histogram_equalization:
    enabled: true
    method: "clahe"
    clahe:
      clip_limit: 2.0      # コントラスト制限値（調整可能）
      tile_grid_size: [8, 8]  # タイルグリッドサイズ（調整可能）
  
  gamma_correction:
    enabled: true
    gamma: 1.2  # ガンマ値（調整可能）
  
  patching:
    enabled: true
    patch_size: [14, 14]  # パッチサイズ（画像サイズに合わせて調整）
    stride: [7, 7]  # ストライド（調整可能）
    padding: true
```

**デモスクリプトで確認**:
```bash
# auguments.yamlの設定を反映したデモを実行
python workspace/demo_preprocessing.py
```

### auguments.yamlの編集（オーグメンテーション）

```yaml
# 画像サイズの変更
image:
  size: [256, 256]  # 224x224 → 256x256

# 学習時のオーグメンテーションを追加
train:
  random_erasing:
    enabled: true  # false → true
    p: 0.3
```

### 前処理の追加

```python
from src.data.preprocessing import (
    HistogramEqualization,
    GammaCorrection,
    PreprocessingPipeline
)

# カスタム前処理パイプライン
preprocessing = PreprocessingPipeline([
    HistogramEqualization(method="clahe"),
    GammaCorrection(gamma=1.2)
])

# Datasetに適用
dataset = ClassificationDataset.from_split_file(
    split_file="artifacts/splits/train_list.txt",
    preprocessing=preprocessing,
    transform=train_transform
)
```

## 📊 データの確認

### データセット情報の確認

```python
dm = ClassificationDataModule(splits_dir="artifacts/splits")
dm.setup("fit")

print(f"学習データ: {len(dm.train_dataset)}サンプル")
print(f"検証データ: {len(dm.val_dataset)}サンプル")
print(f"テストデータ: {len(dm.test_dataset)}サンプル")

# クラス分布
distribution = dm.train_dataset.get_class_distribution()
print(f"クラス分布: {distribution}")
```

### サンプル画像の確認

```python
import matplotlib.pyplot as plt

# 最初のサンプルを取得
image, label = dm.train_dataset[0]

# 画像を表示
plt.imshow(image.permute(1, 2, 0))  # (C, H, W) → (H, W, C)
plt.title(f"Label: {label}")
plt.show()
```

## 🧪 デモとテスト

### デモスクリプトで動作確認（推奨）

デモスクリプトは `auguments.yaml` の設定を読み込んで動作を可視化します。

```bash
# 前処理の動作確認（auguments.yamlのpreprocessing設定を使用）
python workspace/demo_preprocessing.py

# オーグメンテーションの動作確認（auguments.yamlのtrain/val/test設定を使用）
python workspace/demo_augmentation.py

# Dataset/DataModuleの動作確認（auguments.yamlの全設定を使用）
python workspace/demo_dataset.py
```

**重要**: 
- デモスクリプトは本番データ（`data/`）を優先的に使用します
- `data/` がない場合は自動的に `data_for_test/` にフォールバックします
- `auguments.yaml` でパラメータを変更すると、デモの挙動が変わります

### DataModuleのテスト

```bash
# DataModuleを直接実行
python src/data/datamodule.py
```

### 個別コンポーネントのテスト

```bash
# 前処理のテスト
python src/data/preprocessing.py

# オーグメンテーションのテスト
python src/data/augmentation.py
```

## 🔧 トラブルシューティング

### albumentations が見つからない

```bash
pip install albumentations>=1.3.0
```

### opencv-python が見つからない

```bash
pip install opencv-python>=4.8.0
```

### PyTorch Lightning が見つからない

```bash
pip install pytorch-lightning>=2.0.0
```

### 画像が読み込めない

- ファイルパスが正しいか確認
- 画像ファイルが破損していないか確認
- サポートされている形式か確認（JPEG, PNG, etc.）

## 📝 次のステップ

データパイプラインの実装が完了したので、次は**学習レイヤー**の実装に進みます：

1. **モデル定義** (`src/models/`)
   - LightningModule
   - ResNetベースモデル
   
2. **学習スクリプト** (`scripts/train.py`)
   - Trainer設定
   - Callbacks
   - MLflow統合

3. **ハイパーパラメータチューニング** (`scripts/tune.py`)
   - Optuna統合

## 参考資料

- [PyTorch Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
- [albumentations Documentation](https://albumentations.ai/)
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)

