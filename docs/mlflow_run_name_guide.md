# MLflow Run名設定ガイド

MLflowのRun名を`params.yaml`とコマンドライン引数でカスタマイズする方法。

## 📋 概要

Run名を指定することで、MLflow UIで実験を識別しやすくなります。
- **Experiment Name（実験名）**: テーマ名（自動設定）
- **Run Name（Run名）**: カスタマイズ可能（本ガイドの内容）

---

## 🎯 設定方法

### 方法1: params.yamlで設定

**params.yaml**:

```yaml
training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 10
  optimizer: "Adam"
  num_workers: 4
  run_name: "baseline_resnet18"  # ここで指定
```

```bash
# params.yamlのrun_nameが使用される
python scripts/train.py --theme-id 7
```

---

### 方法2: コマンドライン引数で指定

```bash
# コマンドライン引数が優先される（params.yamlを上書き）
python scripts/train.py --theme-id 7 --run-name "experiment_001"
```

---

### 方法3: 未指定（自動生成）

```yaml
# params.yaml
training:
  run_name: null  # または省略
```

```bash
# MLflowが自動的にランダムな名前を生成
python scripts/train.py --theme-id 7
# 例: "jovial-cat-123"
```

---

## 💡 推奨される命名規則

### 1. ベースライン実験
```
baseline_<model_name>
```
**例**:
- `baseline_resnet18`
- `baseline_resnet50`
- `baseline_efficientnet_b0`

---

### 2. データ拡張の実験
```
aug_<technique>
```
**例**:
- `aug_rotation`
- `aug_cutout`
- `aug_mixup`
- `aug_cutmix`
- `aug_combination_v1`

---

### 3. 学習率の調整
```
lr_<value>
```
**例**:
- `lr_0.001`
- `lr_0.01`
- `lr_1e-4`
- `lr_adaptive`

---

### 4. アーキテクチャの変更
```
arch_<architecture>
```
**例**:
- `arch_deeper`
- `arch_wider`
- `arch_attention`
- `arch_custom_v1`

---

### 5. バージョン管理
```
<purpose>_v<number>
```
**例**:
- `baseline_v1`
- `augmentation_v2`
- `production_v3`
- `experiment_v10`

---

### 6. 日付付き
```
<purpose>_YYYYMMDD
```
**例**:
- `baseline_20251117`
- `experiment_20251118`
- `production_20251120`

---

### 7. 組み合わせ
```
<model>_<technique>_<version>
```
**例**:
- `resnet18_aug_rotation_v1`
- `resnet50_lr_0.001_v2`
- `efficientnet_mixup_prod_v3`

---

## 📊 MLflow UIでの確認

### 1. MLflow UIの起動

```bash
cd experiments
mlflow ui --port 5001
```

ブラウザで `http://localhost:5001` にアクセス

---

### 2. UIで確認できる情報

| 項目 | 内容 | 例 |
|------|------|-----|
| **Experiment Name** | テーマ名（自動設定） | "MNIST Test" |
| **Run Name** | 指定したrun名 | "baseline_resnet18" |
| **Parameters** | ハイパーパラメータ | theme_id, theme_name, batch_size など |
| **Metrics** | 学習メトリクス | train_loss, val_accuracy など |
| **Artifacts** | 保存物 | モデル、params.yaml など |

---

## 🚀 実践例

### 例1: ベースライン実験

```bash
# params.yamlで設定
python scripts/train.py --theme-id 7
```

**params.yaml**:
```yaml
training:
  run_name: "baseline_resnet18"
```

---

### 例2: データ拡張の比較

```bash
# 拡張なし
python scripts/train.py --theme-id 7 --run-name "no_aug_v1" --epochs 50

# 回転のみ
python scripts/train.py --theme-id 7 --run-name "aug_rotation_v1" --epochs 50

# 組み合わせ
python scripts/train.py --theme-id 7 --run-name "aug_combination_v1" --epochs 50
```

---

### 例3: 学習率の調整

```bash
python scripts/train.py --theme-id 7 --run-name "lr_0.0001" --learning-rate 0.0001 --epochs 50
python scripts/train.py --theme-id 7 --run-name "lr_0.001" --learning-rate 0.001 --epochs 50
python scripts/train.py --theme-id 7 --run-name "lr_0.01" --learning-rate 0.01 --epochs 50
```

---

### 例4: モデル比較

```bash
# ResNet18
python scripts/train.py --theme-id 7 --run-name "model_resnet18" --epochs 50

# ResNet34
python scripts/train.py --theme-id 7 --run-name "model_resnet34" --epochs 50

# ResNet50
python scripts/train.py --theme-id 7 --run-name "model_resnet50" --epochs 50
```

---

## 🔧 優先順位

Run名の決定は以下の優先順位で行われます：

1. **コマンドライン引数** `--run-name` （最優先）
2. **params.yaml** の `training.run_name`
3. **自動生成** （未指定の場合）

```bash
# 1. コマンドライン引数が最優先
python scripts/train.py --run-name "cli_override"

# 2. params.yamlが使用される
python scripts/train.py  # params.yamlのrun_nameが使用される

# 3. 自動生成
python scripts/train.py  # params.yamlにrun_nameが未設定の場合
```

---

## 📝 ベストプラクティス

### ✅ 推奨

1. **意味のある名前をつける**: 実験の目的が分かる名前
   ```
   ✓ baseline_resnet18
   ✓ aug_rotation_test
   ✓ lr_0.001_experiment
   ```

2. **一貫性のある命名**: プロジェクト全体で統一
   ```
   ✓ baseline_v1, baseline_v2, baseline_v3
   ```

3. **バージョン管理**: 同じ実験の繰り返しにはバージョン番号
   ```
   ✓ experiment_v1, experiment_v2
   ```

---

### ❌ 避けるべき

1. **曖昧な名前**:
   ```
   ✗ test1
   ✗ experiment
   ✗ new
   ```

2. **意味のない文字列**:
   ```
   ✗ abc123
   ✗ temp
   ✗ xxx
   ```

3. **特殊文字**:
   ```
   ✗ baseline@v1
   ✗ test#1
   ✗ exp/001
   ```

---

## 🔍 トラブルシューティング

### Q1: Run名が反映されない

**A**: 以下を確認してください：

1. params.yamlの構文が正しいか
2. コマンドライン引数のスペルが正しいか（`--run-name`）
3. ログを確認（`MLflow run名: 'xxx'`と表示されるか）

---

### Q2: 自動生成された名前を使いたい

**A**: params.yamlで`null`に設定し、コマンドライン引数も指定しない

```yaml
training:
  run_name: null
```

```bash
python scripts/train.py --theme-id 7  # 自動生成される
```

---

### Q3: 既存の実験を同じRun名で実行したい

**A**: MLflowは同じExperiment内で同じRun名を許可します。
各Runは一意のRun IDで識別されるため、同じ名前でも問題ありません。

---

## 📚 関連ドキュメント

- [学習ガイド](training_guide.md)
- [MLflowセットアップ](mlflow_setup.md)
- [進捗管理](progress.md)

---

## 🎓 まとめ

- **Run名は実験の識別に重要**
- **params.yamlまたはコマンドライン引数で指定可能**
- **意味のある命名規則を使用**
- **MLflow UIで簡単に確認できる**

**今すぐ試す**:

```bash
python scripts/train.py --theme-id 7 --run-name "my_first_experiment" --epochs 2 --batch-size 4
```

MLflow UIで確認:

```bash
cd experiments && mlflow ui --port 5001
```

http://localhost:5001 で実験結果を確認してください！

