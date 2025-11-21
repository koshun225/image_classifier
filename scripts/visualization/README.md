# Visualization Scripts（Django統合版）

データの可視化とデバッグのためのスクリプト集です。
DjangoデータベースからテーマIDを指定してデータを取得し、各種可視化を行います。

## 📋 前提条件

1. Djangoデータベースにテーマとデータが登録されていること
2. データ分割（train/valid/test）が実行済みであること
3. `auguments.yaml`が存在すること
4. PyTorch、matplotlib、PIL などがインストールされていること

---

## 📊 スクリプト一覧

### 1. vis_augmentation.py

**目的**: データオーグメンテーションの動作確認

**機能**:
- 学習用オーグメンテーションの可視化（複数サンプル生成）
- 検証/テスト用変換の確認
- albumentations vs torchvision の比較

**使用方法**:

```bash
# 基本的な使用方法
python scripts/visualization/vis_augmentation.py --theme-id 7

# サンプル数を変更
python scripts/visualization/vis_augmentation.py --theme-id 7 --num-samples 16
```

**出力ファイル**:
- `workspace/demo_train_augmentation.png`: 学習用オーグメンテーションのサンプル
- `workspace/demo_val_test_transform.png`: 検証/テスト用変換の比較
- `workspace/demo_library_comparison.png`: ライブラリ比較

**引数**:
| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--theme-id` | ✅ | - | Djangoテーマ ID |
| `--num-samples` | - | 8 | 生成するサンプル数 |

---

### 2. vis_preprocessing.py

**目的**: 前処理の動作確認

**機能**:
- ヒストグラム均等化のデモ（Global / CLAHE）
- ガンマ補正のデモ（複数のガンマ値）
- パッチ化のデモ（パッチ分割の可視化）
- 前処理パイプラインのデモ（統合処理）

**使用方法**:

```bash
# 前処理の可視化
python scripts/visualization/vis_preprocessing.py --theme-id 7
```

**出力ファイル**:
- `workspace/demo_histogram_eq.png`: ヒストグラム均等化
- `workspace/demo_gamma.png`: ガンマ補正
- `workspace/demo_patching.png`: パッチ化
- `workspace/demo_pipeline.png`: 前処理パイプライン

**引数**:
| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--theme-id` | ✅ | - | Djangoテーマ ID |

---

### 3. vis_dataset.py

**目的**: Dataset と DataModule の動作確認

**機能**:
- Datasetの基本動作確認（クラス分布、サンプル形状など）
- Datasetのサンプル可視化（複数画像の表示）
- DataLoaderの動作確認（バッチ処理）
- DataModuleの動作確認（train/valid/test分割）

**使用方法**:

```bash
# Datasetとdatamoduleの可視化
python scripts/visualization/vis_dataset.py --theme-id 7

# サンプル数を変更
python scripts/visualization/vis_dataset.py --theme-id 7 --num-samples 32
```

**出力ファイル**:
- `workspace/demo_dataset_samples.png`: Datasetのサンプル
- `workspace/demo_dataloader_batch.png`: DataLoaderのバッチ
- `workspace/demo_datamodule_samples.png`: DataModuleのサンプル

**引数**:
| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--theme-id` | ✅ | - | Djangoテーマ ID |
| `--num-samples` | - | 16 | 可視化するサンプル数 |

---

## 🚀 使用例

### 基本的なワークフロー

```bash
# 1. テーマIDを確認
python scripts/check_theme_data.py --theme-id 7

# 2. 前処理の確認
python scripts/visualization/vis_preprocessing.py --theme-id 7

# 3. オーグメンテーションの確認
python scripts/visualization/vis_augmentation.py --theme-id 7

# 4. Dataset/DataModuleの確認
python scripts/visualization/vis_dataset.py --theme-id 7
```

### 全てのビジュアライゼーションを実行

```bash
# テーマID 7 で全ての可視化を実行
THEME_ID=7

python scripts/visualization/vis_preprocessing.py --theme-id $THEME_ID
python scripts/visualization/vis_augmentation.py --theme-id $THEME_ID --num-samples 8
python scripts/visualization/vis_dataset.py --theme-id $THEME_ID --num-samples 16

# 生成された画像を確認
ls -lh workspace/demo_*.png
```

---

## 📁 出力ファイル

すべての可視化結果は `workspace/` ディレクトリに保存されます：

```
workspace/
├── demo_train_augmentation.png      # オーグメンテーションサンプル
├── demo_val_test_transform.png      # 検証/テスト変換
├── demo_library_comparison.png      # ライブラリ比較
├── demo_histogram_eq.png             # ヒストグラム均等化
├── demo_gamma.png                    # ガンマ補正
├── demo_patching.png                 # パッチ化
├── demo_pipeline.png                 # 前処理パイプライン
├── demo_dataset_samples.png          # Datasetサンプル
├── demo_dataloader_batch.png         # DataLoaderバッチ
└── demo_datamodule_samples.png       # DataModuleサンプル
```

---

## 🐛 トラブルシューティング

### Q1: "テーマID X が見つかりません"

**A**: Djangoデータベースにテーマが存在するか確認してください。

```bash
# テーマ一覧を確認
python scripts/check_theme_data.py --list-themes
```

---

### Q2: "テーマID X に画像が登録されていません"

**A**: Django管理画面またはWeb UIで画像をアップロードしてください。

---

### Q3: "テーマID X にtrainデータがありません"

**A**: データ分割を実行してください。

```bash
# Django Web UIで実行
# または
python scripts/split_data.py --theme-id 7
```

---

### Q4: "auguments.yaml が見つかりません"

**A**: プロジェクトルートに `auguments.yaml` を配置してください。

```bash
# プロジェクトルートで実行
ls auguments.yaml

# なければテンプレートをコピー
cp config/auguments.yaml.example auguments.yaml
```

---

## 🔧 カスタマイズ

### auguments.yaml の設定

各スクリプトは `auguments.yaml` の設定を使用します：

```yaml
# 画像サイズ
image:
  size: [224, 224]

# データオーグメンテーション
train:
  horizontal_flip:
    enable: true
    p: 0.5
  rotation:
    enable: true
    limit: 15

# 前処理
preprocessing:
  histogram_equalization:
    enable: true
    method: "clahe"
  gamma_correction:
    enable: false
    gamma: 1.2
```

---

## 📚 関連ドキュメント

- [データ管理ガイド](../../docs/data_management_guide.md)
- [学習ガイド](../../docs/training_guide.md)
- [auguments.yaml 設定ガイド](../../docs/augmentation_guide.md)

---

## 🎯 使用シーン

### シーン1: データオーグメンテーションの調整

```bash
# 1. auguments.yamlを編集
vim auguments.yaml

# 2. 可視化して確認
python scripts/visualization/vis_augmentation.py --theme-id 7

# 3. 調整して再確認（繰り返し）
```

---

### シーン2: 前処理パラメータのチューニング

```bash
# 1. 前処理設定を変更
vim auguments.yaml  # preprocessing セクション

# 2. 可視化して効果を確認
python scripts/visualization/vis_preprocessing.py --theme-id 7

# 3. 最適なパラメータを見つける
```

---

### シーン3: Dataset/DataModuleのデバッグ

```bash
# データロードに問題がある場合
python scripts/visualization/vis_dataset.py --theme-id 7

# 出力を確認：
# - サンプル数が正しいか
# - クラス分布が偏っていないか
# - 画像が正しく表示されるか
# - バッチサイズが適切か
```

---

## ✅ まとめ

これらのvisualizationスクリプトは：

- ✅ **Djangoデータベースと完全統合**
- ✅ **theme_idでデータ指定が簡単**
- ✅ **視覚的なデバッグが可能**
- ✅ **設定の調整と確認が効率的**
- ✅ **学習前の事前チェックに最適**

学習を開始する前に、これらのスクリプトでデータとパイプラインを確認することを強く推奨します！

