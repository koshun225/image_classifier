# 学習ガイド

Django学習データ管理システムと統合された学習パイプラインの使用方法。

## 📋 前提条件

### 1. 必要なパッケージのインストール

```bash
# PyTorchのインストール（CPU版）
pip install torch torchvision

# PyTorch Lightningのインストール
pip install pytorch-lightning

# その他の依存パッケージ
pip install pyyaml mlflow albumentations
```

### 2. データの準備

学習を開始する前に、以下が完了していることを確認してください：

1. ✅ Djangoテーマの作成
2. ✅ ラベルの設定
3. ✅ 画像のアップロード
4. ✅ ラベル付け
5. ✅ **データ分割の実行**（重要！）

データ分割の確認：

```python
python scripts/check_theme_data.py --theme-id 7
```

---

## 🚀 学習の実行

### 基本的な使い方

#### 1. デフォルト設定で実行

```bash
python scripts/train.py
```

`params.yaml`のtheme_idに設定されたテーマで学習が実行されます。

#### 2. 特定のテーマIDを指定

```bash
python scripts/train.py --theme-id 7
```

#### 3. パラメータをカスタマイズ

```bash
python scripts/train.py --theme-id 7 \\
  --epochs 50 \\
  --batch-size 32 \\
  --learning-rate 0.001 \\
  --num-workers 4
```

#### 4. MLflow run名を指定

```bash
python scripts/train.py --theme-id 7 \\
  --run-name "baseline_experiment"
```

run名を指定することで、MLflow UIで実験を識別しやすくなります。

#### 5. MLflowなしで軽量実行（テスト用）

```bash
python scripts/train.py --theme-id 7 \\
  --epochs 2 \\
  --batch-size 4 \\
  --no-mlflow
```

---

## ⚙️ 設定ファイル

### params.yaml

学習のハイパーパラメータを設定します。

```yaml
model:
  name: "ResNet18"
  num_classes: 10

training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 10
  optimizer: "Adam"
  num_workers: 4
  run_name: null  # MLflow run名（未指定の場合は自動生成）

data:
  theme_id: 7  # Djangoテーマ ID（必須）
  split_ratio:
    train: 0.7
    valid: 0.15
    test: 0.15
  seed: 42
```

### config.yaml

プロジェクト全体の設定を管理します。

```yaml
mlflow:
  tracking_uri: "experiments/mlruns"
  experiment_name: "classification_with_mlops"

model:
  default_model: "ResNet18"
  available_models:
    - ResNet18
    - ResNet34
    - ResNet50
```

### auguments.yaml

データオーグメンテーションの設定を管理します。

```yaml
image:
  size: [224, 224]

train:
  use_augmentation: true
  augmentations:
    - RandomRotation:
        degrees: 15
    - RandomHorizontalFlip:
        p: 0.5

val:
  use_augmentation: false

test:
  use_augmentation: false
```

---

## 📊 学習の監視

### MLflowを使用した実験管理

学習中、MLflowが自動的にメトリクスとパラメータを記録します。

**実験名は自動的にテーマ名に設定されます**。これにより、複数のテーマで学習を行う場合でも、実験が整理されて管理しやすくなります。

```bash
# MLflow UIを起動
cd experiments
mlflow ui --port 5001

# ブラウザで http://localhost:5001 にアクセス
```

記録される情報：

- **実験名（Experiment Name）**: テーマ名（例: "MNIST Test"）
- **Run名（Run Name）**: 
  - `params.yaml`の`training.run_name`で設定可能
  - コマンドライン引数`--run-name`で上書き可能
  - 未指定の場合は自動生成（例: "jovial-cat-123"）
- **パラメータ**: 
  - theme_id: テーマID
  - theme_name: テーマ名
  - batch_size, learning_rate, epochs など
- **メトリクス**: train_loss, val_loss, val_accuracy, val_f1 など
- **Artifacts**: モデルチェックポイント, params.yaml, confusion matrix など

### Run名の管理

Run名を指定することで、実験の目的や内容を明確に識別できます。

**方法1: params.yamlで設定**

```yaml
training:
  run_name: "baseline_resnet18"
```

**方法2: コマンドライン引数で指定**

```bash
python scripts/train.py --theme-id 7 --run-name "augmentation_test_v1"
```

**推奨される命名規則**:
- `baseline_<model_name>`: ベースライン実験
- `aug_<technique>`: データ拡張の実験
- `lr_<value>`: 学習率の調整
- `arch_<architecture>`: アーキテクチャの変更
- `v1`, `v2`, `v3`: バージョン管理

---

## 🎯 コマンドライン引数一覧

### データ設定

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--theme-id` | Djangoテーマ ID | params.yaml |

### 学習設定

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--epochs` | エポック数 | params.yaml |
| `--batch-size` | バッチサイズ | params.yaml |
| `--learning-rate` | 学習率 | params.yaml |
| `--num-workers` | DataLoaderワーカー数 | params.yaml |

### ファイル設定

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--params` | params.yamlのパス | `params.yaml` |
| `--config` | config.yamlのパス | `config.yaml` |
| `--augments` | auguments.yamlのパス | `auguments.yaml` |

### ディレクトリ設定

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--checkpoint-dir` | チェックポイント保存先 | `checkpoints` |
| `--log-dir` | ログ保存先 | `logs` |

### MLflow設定

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--no-mlflow` | MLflowを無効化 | False |
| `--run-name` | MLflow run名 | 自動生成 |

### トレーニング設定

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--accelerator` | アクセラレータ（cpu, gpu, mps） | `auto` |
| `--devices` | 使用するデバイス | `auto` |
| `--precision` | 精度（32-true, 16-mixed） | `32-true` |
| `--monitor` | モニターするメトリクス | `val_loss` |
| `--use-preprocessing` | 前処理を有効化 | False |

---

## 📖 使用例

### 例1: 軽量テスト実行（2エポック、GPU使用）

```bash
python scripts/train.py \\
  --theme-id 7 \\
  --epochs 2 \\
  --batch-size 8 \\
  --accelerator gpu \\
  --no-mlflow
```

### 例2: 本格的な学習実行（50エポック、MLflow有効）

```bash
python scripts/train.py \\
  --theme-id 7 \\
  --epochs 50 \\
  --batch-size 32 \\
  --learning-rate 0.0001 \\
  --run-name "resnet18_50epochs"
```

### 例3: 前処理を有効にした学習

```bash
python scripts/train.py \\
  --theme-id 7 \\
  --epochs 30 \\
  --use-preprocessing
```

### 例4: M1/M2 Mac（MPS）を使用

```bash
python scripts/train.py \\
  --theme-id 7 \\
  --epochs 20 \\
  --accelerator mps
```

---

## 🔍 トラブルシューティング

### エラー: "theme_idがparams.yamlのdataセクションに設定されていません"

**原因**: params.yamlにtheme_idが設定されていない

**解決策**:

```yaml
# params.yamlに追加
data:
  theme_id: 7  # 使用するテーマID
```

または、コマンドラインで指定：

```bash
python scripts/train.py --theme-id 7
```

### エラー: "No module named 'torch'"

**原因**: PyTorchがインストールされていない

**解決策**:

```bash
# CPU版
pip install torch torchvision

# GPU版（CUDA 11.8の場合）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### エラー: "No module named 'pytorch_lightning'"

**原因**: PyTorch Lightningがインストールされていない

**解決策**:

```bash
pip install pytorch-lightning
```

### エラー: "train/valid/testデータが0枚"

**原因**: データ分割が実行されていない

**解決策**:

```bash
# Django Web UIでデータ分割を実行
# テーマ詳細画面（http://127.0.0.1:8000/theme/<theme_id>/）でデータ分割ボタンをクリック

# または、Pythonスクリプトで実行
python -c "from src.data.split import split_dataset; split_dataset(theme_id=7)"
```

---

## ✅ 学習前のチェックリスト

学習を開始する前に、以下を確認してください：

- [ ] PyTorchとPyTorch Lightningがインストールされている
- [ ] Djangoテーマが作成されている
- [ ] ラベルが設定されている
- [ ] 画像がアップロードされている
- [ ] すべての画像にラベルが付けられている
- [ ] **データ分割が実行されている**（Train/Valid/Testに分割済み）
- [ ] `params.yaml`にtheme_idが設定されている
- [ ] `auguments.yaml`が存在する

確認コマンド：

```bash
python scripts/check_theme_data.py --theme-id 7
```

すべて確認できたら、学習を開始してください！

```bash
python scripts/train.py --theme-id 7
```

---

## 🎓 次のステップ

学習が完了したら：

1. **MLflowで結果を確認**
   ```bash
   cd experiments && mlflow ui
   ```

2. **モデルを登録**（実装予定）
   ```bash
   python scripts/register_model.py --run-id <run_id>
   ```

3. **推論を実行**（実装予定）
   ```bash
   python scripts/predict.py --model-id <model_id> --image <image_path>
   ```

4. **ハイパーパラメータチューニング**
   ```bash
   python scripts/tune.py --theme-id 7
   ```

