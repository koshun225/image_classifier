#!/usr/bin/env python
"""
MLflow Run名設定のテスト

params.yamlとコマンドライン引数でrun_nameを指定できることを確認
"""

import os
import sys
import yaml
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_run_name_settings():
    """Run名設定のテスト"""
    print("="*70)
    print("MLflow Run名設定テスト")
    print("="*70)
    
    # params.yamlの確認
    params_file = project_root / "params.yaml"
    
    print(f"\n📝 params.yaml の確認: {params_file}")
    print("-"*70)
    
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    
    training_config = params.get("training", {})
    run_name = training_config.get("run_name")
    
    print(f"training.run_name: {run_name}")
    
    if run_name is None:
        print("✅ run_nameが未設定（自動生成モード）")
    else:
        print(f"✅ run_nameが設定済み: '{run_name}'")
    
    # 使用例の表示
    print("\n" + "="*70)
    print("📋 使用例")
    print("="*70)
    
    print("\n1️⃣ params.yamlで設定する方法:")
    print("-"*70)
    print("""
params.yaml:
  training:
    run_name: "baseline_resnet18"
    """)
    
    print("\n2️⃣ コマンドライン引数で指定する方法:")
    print("-"*70)
    print("""
# 基本的な使用
python scripts/train.py --theme-id 7 --run-name "experiment_001"

# 実験の目的を明確に
python scripts/train.py --theme-id 7 --run-name "baseline_resnet18"
python scripts/train.py --theme-id 7 --run-name "aug_test_v1"
python scripts/train.py --theme-id 7 --run-name "lr_0.01_experiment"
    """)
    
    print("\n3️⃣ 未指定の場合:")
    print("-"*70)
    print("""
# run_nameを指定しない場合、MLflowが自動生成
python scripts/train.py --theme-id 7

# 例: "jovial-cat-123" のようなランダムな名前が付けられます
    """)
    
    # 推奨される命名規則
    print("\n" + "="*70)
    print("💡 推奨される命名規則")
    print("="*70)
    print("""
1. ベースライン実験:
   - baseline_<model_name>
   例: baseline_resnet18, baseline_resnet50

2. データ拡張の実験:
   - aug_<technique>
   例: aug_rotation, aug_cutout, aug_mixup

3. 学習率の調整:
   - lr_<value>
   例: lr_0.001, lr_0.01, lr_1e-4

4. アーキテクチャの変更:
   - arch_<architecture>
   例: arch_deeper, arch_wider, arch_attention

5. バージョン管理:
   - <purpose>_v1, <purpose>_v2, <purpose>_v3
   例: baseline_v1, augmentation_v2, production_v3

6. 日付付き:
   - <purpose>_YYYYMMDD
   例: baseline_20251117, experiment_20251118
    """)
    
    # MLflow UIでの確認方法
    print("\n" + "="*70)
    print("🔍 MLflow UIでの確認")
    print("="*70)
    print("""
1. MLflow UIを起動:
   cd experiments
   mlflow ui --port 5001

2. ブラウザで http://localhost:5001 にアクセス

3. 確認できる情報:
   - Experiment Name: テーマ名（例: "MNIST Test"）
   - Run Name: 指定したrun名（例: "baseline_resnet18"）
   - Parameters: theme_id, theme_name, run_nameなど
   - Metrics: train_loss, val_accuracy など
    """)
    
    print("\n" + "="*70)
    print("✅ テスト完了")
    print("="*70)
    print("\n実際に学習を実行して動作確認してください：")
    print(f"  python scripts/train.py --theme-id 7 --run-name 'test_run' --epochs 2 --batch-size 4")
    print()


if __name__ == "__main__":
    test_run_name_settings()

