# MLflow セットアップガイド

## MLflowとは

MLflowは実験管理ツールで、以下を記録・管理できます：
- ハイパーパラメータ
- メトリクス（損失、精度など）
- モデル
- Git commit ID、DVCバージョンなどのメタデータ

## セットアップ方法

### 方法1: ローカルファイルのみ（簡単、サーバー不要）

最もシンプルな方法です。サーバーを立ち上げずに学習できます。

#### 1. config.yamlの設定

```yaml
mlflow:
  tracking_uri: experiments/mlruns  # ローカルファイル
  experiment_name: classification_with_mlops
```

#### 2. 学習を実行

```bash
python scripts/train.py --epochs 10
```

学習結果は `experiments/mlruns/` に保存されます。

#### 3. UIで結果を確認

学習後にMLflow UIを起動：

```bash
mlflow ui --backend-store-uri experiments/mlruns

# ブラウザで開く
# http://localhost:5000
```

**メリット:**
- サーバー不要、シンプル
- 学習が早く始められる

**デメリット:**
- 学習中にリアルタイムでUIを見れない
- 複数人での実験管理が難しい

---

### 方法2: MLflow Tracking Server（推奨、高機能）

より高機能な使い方です。サーバーを立ち上げてリアルタイムで実験管理します。

#### 1. MLflow Serverを起動

##### オプションA: スクリプトで起動（推奨）

プロジェクトルートで以下を実行：

```bash
# フォアグラウンドで起動（config.yamlの設定を使用）
./scripts/start_mlflow.sh

# バックグラウンドで起動
./scripts/start_mlflow.sh --background

# ポートを変更して起動
./scripts/start_mlflow.sh --port 5002

# 停止
./scripts/start_mlflow.sh --stop
```

スクリプトは `config.yaml` の `mlflow.server.host` と `mlflow.server.port` を自動的に読み込みます。

##### オプションB: 手動で起動（詳細制御）

ターミナルウィンドウを1つ開いて：

```bash
mlflow server \
  --backend-store-uri sqlite:///experiments/mlflow.db \
  --default-artifact-root experiments/mlruns \
  --host 0.0.0.0 \
  --port 5001
```

このターミナルはMLflow Serverが動いている間は開いたままにします。

バックグラウンドで起動する場合：

```bash
# バックグラウンドで起動
nohup mlflow server \
  --backend-store-uri sqlite:///experiments/mlflow.db \
  --default-artifact-root experiments/mlruns \
  --host 0.0.0.0 \
  --port 5001 > mlflow.log 2>&1 &

# プロセスIDを確認
ps aux | grep mlflow

# 停止する場合
kill <プロセスID>
```

#### 2. config.yamlの設定

```yaml
mlflow:
  tracking_uri: http://localhost:5001  # MLflow Server
  experiment_name: classification_with_mlops
  server:
    host: 0.0.0.0
    port: 5001
```

**注意:**
- `server.host` と `server.port` は `./scripts/start_mlflow.sh` が読み込みます
- `tracking_uri` は学習スクリプト（`train.py`、`tune.py`）が使用します
- ポートは同じ値にしてください

#### 3. ブラウザでUIを開く

```
http://localhost:5001
```

#### 4. 学習を実行

別のターミナルを開いて：

```bash
python scripts/train.py --epochs 10
```

学習中にブラウザでリアルタイムに結果を確認できます。

**メリット:**
- リアルタイムで学習状況を確認
- SQLiteでメタデータを管理（高速）
- 複数の実験を並列実行可能
- チームでの実験管理が容易

**デメリット:**
- サーバーを立ち上げる必要がある

---

## MLflow UI の使い方

### 基本的な見方

1. **Experiments一覧**: 実験の一覧が表示されます
2. **Runs一覧**: 各実験の実行履歴（run）が表示されます
3. **Run詳細**: 
   - **Parameters**: ハイパーパラメータ
   - **Metrics**: 損失、精度などのメトリクス（グラフ表示可能）
   - **Artifacts**: 保存されたファイル（モデル、params.yamlなど）
   - **Tags**: Git commit ID、DVCバージョンなどのメタデータ

### 便利な機能

#### 1. 実験の比較

複数のrunを選択して「Compare」ボタンをクリックすると、並べて比較できます。

#### 2. メトリクスのグラフ化

Metricsタブで、train_lossやval_accなどを時系列グラフで表示できます。

#### 3. パラメータによるフィルタリング

- `params.learning_rate > 0.001` などの条件でrunを絞り込めます

#### 4. モデルのダウンロード

Artifactsタブから学習済みモデルをダウンロードできます。

---

## トラブルシューティング

### MLflow Serverが起動しない

**エラー: Address already in use**

```bash
# ポート5000を使用しているプロセスを確認
lsof -i :5000

# プロセスを終了
kill <プロセスID>

# または、別のポートを使用
mlflow server --port 5001 ...
```

### 学習時に接続エラー

**エラー: Connection refused**

MLflow Serverが起動しているか確認：

```bash
curl http://localhost:5000/health
```

起動していない場合、サーバーを起動してください。

### 実験が表示されない

- ブラウザをリロード（F5）
- `config.yaml`の`tracking_uri`が正しいか確認
- MLflow Serverのログを確認（`mlflow.log`）

---

## 高度な設定

### リモートサーバーでの運用

本番環境では、専用サーバーでMLflowを運用できます：

```bash
# サーバー側
mlflow server \
  --backend-store-uri postgresql://user:password@host:5432/mlflow \
  --default-artifact-root s3://my-bucket/mlruns \
  --host 0.0.0.0 \
  --port 5000

# クライアント側（config.yaml）
mlflow:
  tracking_uri: http://mlflow-server.example.com:5000
  experiment_name: classification_with_mlops
```

### 認証の追加

MLflow Server Proxyを使用して認証を追加できます：

```bash
pip install mlflow-server-proxy

mlflow-server-proxy \
  --backend-store-uri sqlite:///experiments/mlflow.db \
  --default-artifact-root experiments/mlruns \
  --auth basic \
  --username admin \
  --password secret
```

---

## 推奨ワークフロー

### 開発時

1. MLflow Serverを起動（フォアグラウンド）
2. ブラウザでUIを開く（http://localhost:5000）
3. 学習を実行
4. リアルタイムで結果を確認

### 本番運用時

1. MLflow Serverをバックグラウンドで起動
2. 定期的にログを確認（`mlflow.log`）
3. 学習を実行
4. 最良のモデルをModel Registryに登録

---

## 参考リンク

- [MLflow公式ドキュメント](https://www.mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://www.mlflow.org/docs/latest/models.html)

---

## クイックスタート（推奨手順）

```bash
# 1. MLflow Serverを起動（config.yamlの設定を使用）
./scripts/start_mlflow.sh

# 2. 別のターミナルで学習を実行
python scripts/train.py --epochs 2

# 3. ブラウザで結果を確認
# http://localhost:5001
```

これで準備完了です！

