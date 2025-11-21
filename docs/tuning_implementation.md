# チューニング機能のDjango統合実装

## 概要

OptunaによるハイパーパラメータチューニングをDjangoデータベースと統合し、MLflowの親子ラン構造で記録する機能を実装しました。

## 実装内容

### 1. 親子ラン構造

MLflowの親子ラン構造を使用して、チューニング全体と各トライアルを階層的に記録します：

- **親ラン**: `params.yaml`の`run_name`を使用
  - チューニング全体の設定と結果を記録
  - 最良のパラメータとモデルを保存
  - 実験名: Djangoテーマ名

- **子ラン**: 各トライアル（run_name = `trial_{番号}`）
  - トライアルごとのパラメータとメトリクスを記録
  - 各トライアルのモデルを保存
  - 親ランにネストされる

### 2. モデル登録

- **best trialのみ**をMLflowモデルレジストリに登録
- モデル名: `{theme_name}_model`
- 親ランにも最良モデルをコピー

### 3. Django統合

- `--theme-id`引数でテーマIDを指定
- Djangoデータベースからトレーニングデータを取得
- テーマ名をMLflow実験名として使用

## 使用方法

### 基本的な使い方

```bash
# theme_idを指定してチューニング実行
python scripts/tune.py --theme-id 7
```

### その他のオプション

```bash
# トライアル数を指定
python scripts/tune.py --theme-id 7 --n-trials 10

# Optunaストレージを使用（複数プロセスでの並列実行用）
python scripts/tune.py --theme-id 7 --storage sqlite:///optuna.db --study-name my_study

# タイムアウト（秒）を指定
python scripts/tune.py --theme-id 7 --timeout 3600

# 設定ファイルを指定
python scripts/tune.py --theme-id 7 --params params.yaml --config config.yaml
```

### params.yamlの設定

チューニング実行前に、`params.yaml`に親ランの名前を設定します：

```yaml
training:
  run_name: "my_tuning_experiment"  # 親ラン名
  # その他のパラメータ...

data:
  theme_id: 7  # コマンドラインで上書き可能
```

### optuna_params.yamlの設定

チューニング対象のパラメータと探索範囲を設定します：

```yaml
study:
  name: "classification_study"
  direction: "maximize"  # または "minimize"
  metric: "test_acc"  # 最適化するメトリクス名
  n_trials: 50
  timeout: null

params_to_tune:
  training.learning_rate:
    type: "float"
    low: 0.0001
    high: 0.01
    log: true
    
  training.batch_size:
    type: "categorical"
    choices: [2, 4, 8, 18, 36]  # 選択肢から選ばれる
    
  training.optimizer:
    type: "categorical"
    choices: ["Adam", "SGD", "AdamW"]
    
  model.name:
    type: "categorical"
    choices: ["ResNet18", "ResNet50", "ResNet101"]

base_params:  # 固定パラメータ
  model:
    num_classes: 10
  training:
    num_epochs: 3  # エポック数
    run_name: "optuna_tuning"  # MLflow親ラン名（params.yamlに無い場合に使用）
    num_workers: 0  # DataLoaderのワーカー数
  data:
    split_ratio:
      train: 0.7
      valid: 0.15
      test: 0.15
    seed: 42
  preprocessing:
    brightness_correction:
      enabled: true
      method: "histogram_equalization"
    patch:
      enabled: false
```

**重要な設定項目：**

- **params_to_tune**: チューニング対象のハイパーパラメータ
  - `type: "float"`: 連続値（low, high, logを指定）
  - `type: "int"`: 整数値（low, high, stepを指定）
  - `type: "categorical"`: 選択肢から選ぶ（choicesを指定）

- **base_params**: 全トライアルで固定のパラメータ
  - `training.num_epochs`: エポック数（固定値）
  - `training.run_name`: MLflow親ラン名（params.yamlより優先度低）
  - その他のトレーニング設定

## MLflowでの確認

チューニング実行後、MLflow UIで結果を確認できます：

```bash
# MLflow UIを起動
bash scripts/start_mlflow.sh

# ブラウザで http://localhost:5001 を開く
```

### 確認項目

1. **実験一覧**: テーマ名の実験を選択
2. **親ラン**: `params.yaml`で設定した`run_name`のランを確認
   - Parameters: チューニング設定
   - Metrics: best_value（最良トライアルのスコア）
   - Artifacts: 設定ファイル、最良パラメータ、最良モデル
3. **子ラン**: 親ランの下に各トライアルが表示
   - Parameters: 各トライアルのハイパーパラメータ
   - Metrics: テスト結果（test_acc, test_lossなど）
   - Artifacts: モデル、設定ファイル

## ディレクトリ構造

```
.
├── scripts/
│   └── tune.py                     # チューニング実行スクリプト
├── src/
│   └── tuning/
│       └── optuna_tuner.py         # Optunaチューナー（親子ラン対応）
├── optuna_params.yaml              # チューニング設定
├── params.yaml                     # 学習パラメータ（run_name含む）
├── config.yaml                     # 全体設定
└── auguments.yaml                  # データ拡張設定
```

## 主な変更点

### src/tuning/optuna_tuner.py

1. **objective関数**
   - `parent_run_id`を受け取る
   - 子ランを作成してトライアルを実行
   - enable_mlflow=Falseで学習し、手動でメトリクスとモデルをログ

2. **tune関数**
   - `params_file`引数を追加（run_name取得用）
   - 親ランを作成
   - 各トライアルを子ランとして実行
   - best trialのモデルを登録

### scripts/tune.py

1. **引数の追加**
   - `--theme-id`: テーマIDの指定
   - `--params`: params.yamlファイルのパス

2. **override_params関数**
   - theme_idをparams.yamlに上書き

3. **結果表示の拡張**
   - 親ランID、best trialのMLflow run IDを表示

## トラブルシューティング

### theme_idが見つからない

```
ValueError: テーマID 7 が見つかりません
```

→ Djangoデータベースにテーマが登録されているか確認してください。

### メトリクスが見つからない

```
ValueError: メトリクス test_acc が見つかりません
```

→ `optuna_params.yaml`の`metric`フィールドが正しいか確認してください。
→ train関数が返すtest_resultsに該当するメトリクスが含まれているか確認してください。

### モデルがログされない

```
WARNING: Trial X: チェックポイントが見つかりません
```

→ train関数がチェックポイントを正しく保存しているか確認してください。
→ `checkpoint_dir`が存在するか確認してください。

## 今後の拡張

以下の機能は今後の実装予定です：

1. **Django Web UI**
   - チューニング設定画面
   - 進捗監視画面
   - トライアル一覧表示
   - 最良パラメータの表示と適用

2. **非同期実行**
   - バックグラウンドでのチューニング実行
   - 進捗状況のリアルタイム更新

3. **データベースモデル**
   - TuningStudyモデル
   - TuningTrialモデル
   - TuningParamConfigモデル

4. **可視化**
   - Optuna風の最適化履歴グラフ
   - パラメータ重要度グラフ
   - パラレル座標プロット

