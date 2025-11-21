# 再現性の確保 (Reproducibility)

> **⚠️ 重要な注意事項**
> 
> 本ドキュメントは**旧ファイルベース実装**（v1/v2/v3フォルダ、artifacts/splits/）の再現性について説明しています。
> 
> **現在のプロジェクトは Djangoデータベースベースに完全移行しました。**
> 
> **Djangoベースの再現性：**
> - データ分割情報：Djangoデータベース（`database.db`）で管理
> - 学習データ：テーマIDで管理
> - 再現性：Git commit ID + MLflow run_id + データベースで保証
> 
> 詳細は以下を参照してください：
> - `docs/django_setup.md`: Django環境のセットアップ
> - `docs/training_guide.md`: 学習の実行方法
> - `docs/mlflow_setup.md`: MLflowによるバージョン管理
> 
> 以下は参考資料として残しています。

---

## 【参考】旧ファイルベース実装の再現性

### split_config.yaml（廃止）

`scripts/split_data.py`（削除済み）を実行すると、自動的に `artifacts/splits/split_config.yaml`（廃止）が生成されました。

**記録される情報:**

```yaml
created_at: '2025-11-12T22:00:31.279512'
data:
  data_dir: data                    # 使用したデータディレクトリ
  num_classes: 10                   # クラス数
  total_samples: 1000               # 総サンプル数
  versions:                         # 使用したデータバージョン
  - ver1
  - ver2
  - ver3
split:
  seed: 42                          # ランダムシード
  test_ratio: 0.15                  # テストデータの割合
  train_ratio: 0.7                  # 訓練データの割合
  valid_ratio: 0.15                 # 検証データの割合
```

### config.yaml の last_used セクション

`--update-config` オプションを使用すると、`config.yaml` にも最後に使用したデータセット情報が記録されます。

```yaml
data:
  root_dir: data
  test_dir: data_for_test
  last_used:                        # 最後に使用したデータセット
    data_dir: data
    versions:
    - ver1
    - ver2
    - ver3
    updated_at: '2025-11-12T22:00:31.283109'
```

**使用方法:**

```bash
# 通常の使用（config.yamlも自動更新される）
python scripts/split_data.py

# config.yamlの更新をスキップする場合
python scripts/split_data.py --no-update-config
```

## データバージョニング (DVC)

### DVCによるデータ追跡

データとアーティファクトはDVCで管理されます：

```bash
# データをDVCに追加
python scripts/manage_dvc.py full --push

# .dvcファイルをGitにコミット
git add *.dvc .gitignore
git commit -m 'Add data v1.0'
git push
```

### データの復元

```bash
# 特定のコミットに戻る
git checkout <commit-hash>

# DVCでデータを復元
dvc pull
```

## 実験の再現手順

### 1. 環境の準備

```bash
# リポジトリをクローン
git clone <repository-url>
cd classification_with_mlops

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 2. データの復元

```bash
# DVCリモートを設定（初回のみ）
dvc remote add -d myremote <remote-url>

# データを取得
dvc pull
```

### 3. データ分割の再現

**方法1: split_config.yaml を使用**

```bash
# split_config.yamlから設定を読み込んで再現
cd artifacts/splits
cat split_config.yaml  # 設定を確認

# 同じ設定で分割を再実行
cd ../..
python scripts/split_data.py \
  --data-dir data \
  --versions ver1 ver2 ver3 \
  --force
```

**方法2: config.yaml の last_used を使用**

```bash
# config.yamlに記録された設定を使用
python scripts/split_data.py --update-config
```

**方法3: DVC で分割ファイルを復元**

```bash
# 分割ファイルもDVCで管理している場合
dvc pull
```

### 4. 学習の再現

```bash
# params.yamlの設定を使用して学習
python scripts/train.py

# または特定のパラメータで学習
python scripts/train.py --epochs 100 --batch-size 32
```

## 再現性チェックリスト

実験を記録する際は、以下を確認してください：

- [ ] `split_config.yaml` が作成されている
- [ ] データとアーティファクトが DVC で管理されている
- [ ] `.dvc` ファイルが Git にコミットされている
- [ ] `params.yaml` の設定が記録されている
- [ ] 実験結果が MLflow に記録されている
- [ ] モデルが MLflow Model Registry に登録されている

## トラブルシューティング

### split_config.yaml が見つからない

```bash
# 新しいバージョンのスクリプトで再実行
python scripts/split_data.py --update-config
```

### DVCでデータが復元できない

```bash
# リモート設定を確認
dvc remote list

# リモートを再設定
dvc remote add -d myremote <remote-url>

# 再度プル
dvc pull
```

### 分割結果が一致しない

- `split_config.yaml` の `seed` を確認
- `params.yaml` の `data.seed` を確認
- 使用したデータバージョンを確認（`split_config.yaml` の `versions`）

## ベストプラクティス

1. **デフォルトで config.yaml が更新される**
   ```bash
   # 通常はこれだけでOK（自動的に記録される）
   python scripts/split_data.py
   
   # 更新したくない場合のみ
   python scripts/split_data.py --no-update-config
   ```

2. **変更をコミットする前に確認**
   ```bash
   git diff config.yaml
   cat artifacts/splits/split_config.yaml
   ```

3. **DVCとGitを同時にコミット**
   ```bash
   # データを追加
   python scripts/manage_dvc.py full --push
   
   # 設定と.dvcファイルをコミット
   git add config.yaml params.yaml *.dvc .gitignore artifacts/splits/split_config.yaml
   git commit -m 'Experiment: Add ver3 dataset'
   git push
   ```

4. **実験ノートを記録**
   - 実験の目的
   - 使用したデータバージョン
   - 変更した設定
   - 結果と考察

## 関連ドキュメント

- [config.yaml の説明](config.md)
- [DVCの使い方](../scripts/README.md#manage_dvcpy)
- [データ分割の説明](../scripts/README.md#split_datapy)

