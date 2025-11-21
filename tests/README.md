# テストディレクトリ

このディレクトリには、プロジェクトのテストコードが含まれています。

## テスト一覧

### test_split_consistency.py

データ分割の一貫性を検証するテストです。

### test_pipeline_integration.py

データセット読み込みからモデル出力までの統合テストです。

**目的:**
- データセットの読み込みが正常に動作することを検証
- DataModuleのセットアップとDataLoaderの動作確認
- モデルの作成と順伝播の動作確認
- LightningModuleのtraining_stepの動作確認
- エンドツーエンドでエラーが発生しないことを保証

**実行方法:**

pytestを使用して実行（推奨）:
```bash
# すべてのテストを実行
pytest tests/test_pipeline_integration.py -v

# 特定のテストクラスのみ実行
pytest tests/test_pipeline_integration.py::TestEndToEndPipeline -v

# 詳細な出力で実行
pytest tests/test_pipeline_integration.py -v -s

# 統合テストのみ実行（エンドツーエンド）
pytest tests/test_pipeline_integration.py::TestEndToEndPipeline::test_full_pipeline -v -s
```

**テストの構成:**

1. **TestDatasetLoading**: データセット読み込みのテスト
   - `test_dataset_from_split_file`: 分割ファイルからDatasetを作成
   - `test_dataset_with_transforms`: 変換を適用したDatasetの動作確認

2. **TestDataModule**: DataModuleのテスト
   - `test_datamodule_setup`: DataModuleのセットアップ確認
   - `test_datamodule_dataloaders`: DataLoaderの動作確認

3. **TestModelCreation**: モデル作成のテスト
   - `test_create_model`: 各種ResNetモデルの作成確認（パラメータ化テスト）
   - `test_model_forward`: モデルの順伝播確認

4. **TestLightningModule**: LightningModuleのテスト
   - `test_lightning_module_creation`: LightningModuleの作成確認
   - `test_lightning_module_forward`: LightningModuleの順伝播確認
   - `test_training_step`: training_stepの動作確認

5. **TestEndToEndPipeline**: エンドツーエンドテスト
   - `test_full_pipeline`: データ読み込み→モデル出力までの全体の流れ
   - `test_full_pipeline_with_preprocessing`: 前処理を含む全体の流れ

**期待される結果:**
```
tests/test_pipeline_integration.py::TestDatasetLoading::test_dataset_from_split_file PASSED
tests/test_pipeline_integration.py::TestDataModule::test_datamodule_setup PASSED
tests/test_pipeline_integration.py::TestModelCreation::test_create_model[ResNet18] PASSED
tests/test_pipeline_integration.py::TestLightningModule::test_lightning_module_creation PASSED
tests/test_pipeline_integration.py::TestEndToEndPipeline::test_full_pipeline PASSED
✓ 統合テスト成功: num_classes=10, batch_size=4, loss=2.3456

すべてのテスト: 15 passed
```

**前提条件:**
- `artifacts/splits/`に分割ファイル（train_list.txt, valid_list.txt, test_list.txt）が存在すること
- `scripts/split_data.py`を実行してデータ分割を準備してください

**注意:**
- テストは実際のデータセットを使用します
- モデルは`pretrained=False`で高速化されています
- `num_workers=0`でテストを実行します（並列処理なし）

**目的:**
- ver1, ver2のみでsplitした結果と、ver1, ver2, ver3でsplitした結果を比較
- ver1, ver2のデータが同じsplit（train/valid/test）に分類されているかを検証
- データ分割の一貫性保証機能が正しく動作していることを確認

**実行方法:**

pytestを使用して実行（推奨）:
```bash
# すべてのテストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_split_consistency.py

# 詳細な出力で実行
pytest tests/test_split_consistency.py -v

# カバレッジレポート付きで実行
pytest tests/test_split_consistency.py --cov=src --cov-report=html
```

直接実行する場合:
```bash
python tests/test_split_consistency.py
```

**テストの流れ:**
1. **DatasetA作成**: ver1, ver2のみでデータ分割を実行
2. **DatasetB作成**: ver1, ver2, ver3を統合してデータ分割を実行（既存の分割情報を使用）
3. **一貫性検証**: ver1, ver2の画像が両方のdatasetで同じsplitに属しているかを検証

**期待される結果:**
```
TRAIN: ✓ PASS
  一貫性あり: 460/460枚 (100.0%)
  一貫性なし: 0/460枚 (0.0%)

VALID: ✓ PASS
  一貫性あり: 90/90枚 (100.0%)
  一貫性なし: 0/90枚 (0.0%)

TEST: ✓ PASS
  一貫性あり: 110/110枚 (100.0%)
  一貫性なし: 0/110枚 (0.0%)

✓ すべてのテストが成功しました！
データ分割の一貫性が保証されています。
```

**前提条件:**
- `data_for_test/ver1/`, `data_for_test/ver2/`, `data_for_test/ver3/`にテストデータが準備されていること
- `workspace/mnist_data_prepare.py`を実行してテストデータを準備してください

**注意:**
- テストは`data_for_test/`ディレクトリを使用します
- 本番用の`data/`ディレクトリには影響しません
- `config.yaml`でデータディレクトリのパスを管理しています

## データ分割の一貫性保証とは

複数のデータセットバージョンを段階的に追加する際、既存のデータの分割を維持することで、以下を保証します：

1. **データリークの防止**: 
   - ver1でtrainに分類されたデータは、ver1+ver2統合時もtrainに分類
   - ver1でvalid/testに分類されたデータは、統合後もvalid/testに分類

2. **評価の公平性**: 
   - テストデータが学習データに混入しない
   - 一貫した評価基準を維持

3. **再現性**: 
   - 同じデータセットバージョンの組み合わせであれば、同じ分割結果を得られる

## 実装の詳細

このテストは、`src/data/split.py`の`split_dataset_with_consistency()`関数を使用しています。

**関数の動作:**
1. 既存の分割情報を読み込む
2. 新しいデータセットを読み込む
3. 既存の分割にあるファイルはその分割を維持
4. 新しいファイルのみを指定の比率で分割

**コード例:**
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

## Pytestの機能

### テストの分類

テストは以下のように分類されています：

1. **test_data_directories_exist**: データディレクトリの存在確認
2. **test_train_split_consistency**: trainデータの一貫性検証
3. **test_valid_split_consistency**: validデータの一貫性検証
4. **test_test_split_consistency**: testデータの一貫性検証
5. **test_all_files_accounted_for**: すべてのファイルの整合性確認

### Fixture

- **dataset_a_splits**: ver1, ver2のみの分割結果（module scope）
- **dataset_b_splits**: ver1, ver2, ver3統合の分割結果（module scope）
- **params**: params.yamlの設定（session scope）
- **data_dir**: データディレクトリのパス（session scope）

### 実行オプション

```bash
# 特定のテストのみ実行
pytest tests/test_split_consistency.py::test_train_split_consistency

# 失敗したテストのみ再実行
pytest --lf

# 並列実行（pytest-xdist必要）
pytest -n auto

# マーカーを使用したテスト選択
pytest -m consistency
```

## トラブルシューティング

### テストが失敗する場合

**症状:** 一貫性なしのファイルが多数報告される

**原因:** `split_dataset_with_consistency()`関数が使用されていない可能性があります

**解決方法:**
1. `tests/test_split_consistency.py`を確認
2. `create_split_for_versions()`関数で`existing_splits`引数が正しく渡されているか確認
3. `split_dataset_with_consistency()`が呼び出されているか確認

### データが見つからない

**症状:** `エラー: data/ver1ディレクトリが存在しません`

**解決方法:**
```bash
# MNISTデータを準備
python workspace/mnist_data_prepare.py
```

## 今後の拡張

- 他のデータセットバージョンの組み合わせでのテスト
- パフォーマンステスト（大規模データセット）
- エッジケースのテスト（空のクラス、極端な分割比率など）

