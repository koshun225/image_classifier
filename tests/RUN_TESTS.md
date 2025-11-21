# テスト実行ガイド

## セットアップ

### 1. 依存パッケージのインストール

```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. データの準備

統合テストを実行する前に、データセットと分割ファイルを準備します。

```bash
# データセットの準備（DVCを使用）
dvc pull

# または、テストデータを準備
python workspace/mnist_data_prepare.py

# データ分割の実行
python scripts/split_data.py --data-dir data_for_test --versions ver1 ver2 ver3
```

## テストの実行

### すべてのテストを実行

```bash
pytest tests/ -v
```

### 統合テストのみ実行

```bash
pytest tests/test_pipeline_integration.py -v
```

### 特定のテストクラスのみ実行

```bash
# データセット読み込みのテスト
pytest tests/test_pipeline_integration.py::TestDatasetLoading -v

# DataModuleのテスト
pytest tests/test_pipeline_integration.py::TestDataModule -v

# モデル作成のテスト
pytest tests/test_pipeline_integration.py::TestModelCreation -v

# LightningModuleのテスト
pytest tests/test_pipeline_integration.py::TestLightningModule -v

# エンドツーエンドテスト
pytest tests/test_pipeline_integration.py::TestEndToEndPipeline -v
```

### 詳細な出力で実行

```bash
pytest tests/test_pipeline_integration.py -v -s
```

### カバレッジレポート付きで実行

```bash
pytest tests/test_pipeline_integration.py --cov=src --cov-report=html
```

## トラブルシューティング

### ModuleNotFoundError: No module named 'torch'

**原因:** PyTorchがインストールされていません。

**解決方法:**
```bash
pip install -r requirements.txt
```

### 分割ファイルが見つかりません

**原因:** `artifacts/splits/`に分割ファイルが存在しません。

**解決方法:**
```bash
python scripts/split_data.py
```

### データディレクトリが存在しません

**原因:** データセットが準備されていません。

**解決方法:**
```bash
# DVCを使用する場合
dvc pull

# テストデータを使用する場合
python workspace/mnist_data_prepare.py
python scripts/split_data.py --data-dir data_for_test --versions ver1 ver2 ver3
```

## テスト結果の例

```
tests/test_pipeline_integration.py::TestDatasetLoading::test_dataset_from_split_file PASSED
tests/test_pipeline_integration.py::TestDatasetLoading::test_dataset_with_transforms PASSED
tests/test_pipeline_integration.py::TestDataModule::test_datamodule_setup PASSED
tests/test_pipeline_integration.py::TestDataModule::test_datamodule_dataloaders PASSED
tests/test_pipeline_integration.py::TestModelCreation::test_create_model[ResNet18] PASSED
tests/test_pipeline_integration.py::TestModelCreation::test_create_model[ResNet34] PASSED
tests/test_pipeline_integration.py::TestModelCreation::test_create_model[ResNet50] PASSED
tests/test_pipeline_integration.py::TestModelCreation::test_model_forward PASSED
tests/test_pipeline_integration.py::TestLightningModule::test_lightning_module_creation PASSED
tests/test_pipeline_integration.py::TestLightningModule::test_lightning_module_forward PASSED
tests/test_pipeline_integration.py::TestLightningModule::test_training_step PASSED
tests/test_pipeline_integration.py::TestEndToEndPipeline::test_full_pipeline PASSED
✓ 統合テスト成功: num_classes=10, batch_size=4, loss=2.3456
tests/test_pipeline_integration.py::TestEndToEndPipeline::test_full_pipeline_with_preprocessing PASSED
✓ 前処理付き統合テスト成功: num_classes=10, batch_size=4

================================ 13 passed in 12.34s ================================
```

## CI/CD統合

GitHub ActionsなどのCI/CDパイプラインに統合する場合:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 推奨事項

1. **定期的なテスト実行**: コードを変更するたびにテストを実行
2. **カバレッジの確認**: テストカバレッジを80%以上に保つ
3. **CI/CD統合**: 自動テストをCI/CDパイプラインに組み込む
4. **テストの追加**: 新機能を追加する際は対応するテストも追加

