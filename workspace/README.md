# Workspace - データ準備とテスト

このディレクトリには、データ準備とテスト用のスクリプトが含まれています。

## スクリプト一覧

### 1. mnist_data_prepare.py

MNISTデータをダウンロードし、3つのバージョン（ver1, ver2, ver3）に分割して配置します。

**実行方法:**
```bash
python workspace/mnist_data_prepare.py
```

**処理内容:**
- MNISTデータセット（訓練データ）をダウンロード
- 各クラス（0-9）から100枚ずつサンプリング（合計1,000枚）
- 3つのバージョンに分割:
  - `data_for_test/ver1/`: 各クラス約33枚（合計約330枚）
  - `data_for_test/ver2/`: 各クラス約33枚（合計約330枚）
  - `data_for_test/ver3/`: 各クラス約34枚（合計約340枚）

**注意:** このスクリプトはテスト専用データを`data_for_test/`に保存します。本番用データは`data/`ディレクトリに配置してください。

**必要な依存関係:**
```bash
pip install torchvision
```

#### test_split.py

データセットを読み込み、train/valid/testに分割してテストします。

**実行方法:**
```bash
python workspace/test_split.py
```

**処理内容:**
1. `params.yaml`から分割設定を読み込み
2. **`data/`ディレクトリ下のすべてのデータセットディレクトリを自動検出して統合**
   - 例: `ver1`, `ver2`, `ver3` や `dataset1`, `dataset2` など、任意の名前に対応
   - クラスフォルダ（サブディレクトリ）を持つディレクトリのみを検出
3. train/valid/testに分割（デフォルト: 70%/15%/15%）
4. 分割情報を`artifacts/splits/`に保存:
   - `train_list.txt`（全バージョンのtrainデータ）
   - `valid_list.txt`（全バージョンのvalidデータ）
   - `test_list.txt`（全バージョンのtestデータ）
5. バージョン別の内訳を表示

**必要な依存関係:**
```bash
pip install pyyaml
```

## 使用例

### ステップ1: MNISTデータの準備

```bash
# MNISTデータをダウンロードして配置
python workspace/mnist_data_prepare.py
```

実行後、以下のディレクトリ構造が作成されます:
```
data/
├── ver1/
│   ├── class0/
│   ├── class1/
│   ├── ...
│   └── class9/
├── ver2/
│   └── ...
└── ver3/
    └── ...
```

### ステップ2: データ分割のテスト

```bash
# データを読み込んでtrain/valid/testに分割
python workspace/test_split.py
```

実行後、以下のファイルが作成されます:
```
artifacts/splits/
├── train_list.txt
├── valid_list.txt
└── test_list.txt
```

各ファイルの形式:
```
/path/to/image.png    class_name
/path/to/image2.png   class_name
...
```

## 分割設定のカスタマイズ

`params.yaml`を編集することで、分割比率を変更できます:

```yaml
data:
  split_ratio:
    train: 0.7    # 訓練データ: 70%
    valid: 0.15   # 検証データ: 15%
    test: 0.15    # テストデータ: 15%
  seed: 42        # ランダムシード（再現性のため）
```

## 複数バージョンの統合

`test_split.py`は、`data/`ディレクトリ下のすべてのデータセットディレクトリを自動検出します。

**対応するディレクトリ構造の例:**

```
data/
├── ver1/              # 自動検出
│   ├── class0/
│   ├── class1/
│   └── ...
├── ver2/              # 自動検出
│   └── ...
└── ver3/              # 自動検出
    └── ...
```

または

```
data/
├── dataset1/          # 自動検出
├── dataset2/          # 自動検出
├── dataset3/          # 自動検出
└── dataset4/          # 自動検出
```

プログラム内で手動で統合する場合は、`src/data/dataset.py`の`load_multiple_versions()`関数を使用します:

```python
from src.data.dataset import load_multiple_versions

# 任意の名前のデータセットを統合
class_to_files = load_multiple_versions("data", ["dataset1", "dataset2", "dataset3"])
```

### データ分割の一貫性保証

複数バージョンを統合する際、既存の分割情報を考慮して新しいデータを分割することができます:

```python
from src.data.split import split_dataset_with_consistency, load_split_info

# 既存の分割情報を読み込み
existing_splits = load_split_info("artifacts/splits")

# 一貫性を保証しながら分割
train_data, valid_data, test_data = split_dataset_with_consistency(
    class_to_files,
    existing_splits=existing_splits,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42
)
```

これにより、ver1でtrainに分類されたデータは、ver1+ver2の統合データでもtrainに分類されます。

## トラブルシューティング

### NumPyのバージョン警告が表示される

```bash
# NumPy 1.xにダウングレード
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### データディレクトリが存在しない

```bash
# dataディレクトリが存在することを確認
ls data/

# 存在しない場合は作成
mkdir -p data/ver1 data/ver2 data/ver3
```

### torchvisionがインストールされていない

```bash
pip install torchvision
```


---

## デモスクリプト

実装したコンポーネントの動作確認用デモスクリプトです。

### demo_preprocessing.py

前処理モジュールの動作確認デモです。

**機能:**
- ヒストグラム均等化のデモ（Global / CLAHE）
- ガンマ補正のデモ（異なるガンマ値）
- パッチ化のデモ
- 前処理パイプラインのデモ

**重要**: このデモは`auguments.yaml`の`preprocessing`セクションから設定を読み込みます。前処理のパラメータ（パッチサイズ、ガンマ値、CLAHE設定など）を調整したい場合は、`auguments.yaml`を編集してください。

**使用方法:**

```bash
# 前提: data_for_test/が存在すること
python scripts/visualization/vis_preprocessing.py
```

**設定例（auguments.yaml）:**

```yaml
# auguments.yaml
preprocessing:
  histogram_equalization:
    enabled: true
    method: "clahe"
    clahe:
      clip_limit: 2.0
      tile_grid_size: [8, 8]
  
  gamma_correction:
    enabled: true
    gamma: 1.2
  
  patching:
    enabled: true
    patch_size: [14, 14]  # MNIST(28x28)用
    stride: [7, 7]
    padding: true
```

**生成される画像:**
- `workspace/demo_histogram_eq.png`: ヒストグラム均等化の比較
- `workspace/demo_gamma.png`: ガンマ補正の比較
- `workspace/demo_patching.png`: パッチ化の結果
- `workspace/demo_pipeline.png`: パイプライン適用結果

### demo_augmentation.py

データオーグメンテーションの動作確認デモです。

**機能:**
- 学習用オーグメンテーションのデモ（同じ画像に複数回適用）
- 検証/テスト用変換のデモ
- albumentations と torchvision の比較

**使用方法:**

```bash
python scripts/visualization/vis_augmentation.py
```

**生成される画像:**
- `workspace/demo_train_augmentation.png`: 学習用拡張の例
- `workspace/demo_val_test_transform.png`: 検証/テスト用変換
- `workspace/demo_library_comparison.png`: ライブラリ比較

### demo_dataset.py

Dataset と DataModule の動作確認デモです。

**機能:**
- Dataset の基本動作確認
- Dataset サンプルの可視化
- DataLoader の動作確認
- DataModule の動作確認

**使用方法:**

```bash
# 前提: artifacts/splits/が存在すること
python scripts/visualization/vis_dataset.py
```

**生成される画像:**
- `workspace/demo_dataset_samples.png`: Datasetサンプル
- `workspace/demo_dataloader_batch.png`: DataLoaderバッチ
- `workspace/demo_datamodule_samples.png`: DataModuleサンプル

---

### 全デモの実行手順

**本番データで実行する場合:**

```bash
# 1. 本番データを配置
# data/ver1/, data/ver2/ などのディレクトリにクラスフォルダ構造でデータを配置

# 2. データ分割（本番データ）
python scripts/split_data.py

# 3. デモスクリプト実行
python scripts/visualization/vis_preprocessing.py    # auguments.yamlの前処理設定を確認
python scripts/visualization/vis_augmentation.py     # auguments.yamlのオーグメンテーション設定を確認
python scripts/visualization/vis_dataset.py          # Dataset/DataModuleの動作確認
```

**テスト用データで実行する場合:**

```bash
# 1. テスト用データ準備（MNIST）
python workspace/mnist_data_prepare.py

# 2. データ分割（テスト用）
python scripts/split_data.py --data-dir data_for_test --output-dir artifacts/splits

# 3. デモスクリプト実行（data/ がない場合は自動的に data_for_test/ を使用）
python scripts/visualization/vis_preprocessing.py
python scripts/visualization/vis_augmentation.py
python scripts/visualization/vis_dataset.py
```

生成された画像は `workspace/` ディレクトリに保存されます（`.gitignore`で除外済み）。

**重要**: 
- デモスクリプトは `auguments.yaml` の設定を読み込んで動作します
- 前処理やオーグメンテーションのパラメータを変更したい場合は `auguments.yaml` を編集してください
- 本番データ（`data/`）が存在する場合は優先的に使用し、なければ `data_for_test/` を使用します
