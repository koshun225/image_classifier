"""
前処理の動作確認デモ

前処理モジュールの各機能を可視化して確認します。
Djangoデータベースからテーマの画像を取得して使用します。
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Django環境のセットアップ
sys.path.insert(0, str(project_root / 'src' / 'web'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from src.data.preprocessing import (
    HistogramEqualization,
    GammaCorrection,
    Patching,
    Resize,
    PreprocessingPipeline,
    create_preprocessing_pipeline,
    create_patching,
    load_image
)
from data_management.crud import get_theme, get_traindata_by_theme


def demo_histogram_equalization(image_path: str, config: dict = None):
    """ヒストグラム均等化のデモ"""
    print("\n" + "=" * 60)
    print("ヒストグラム均等化のデモ")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    
    # 設定からパラメータを取得
    if config:
        hist_eq_config = config.get("histogram_equalization", {})
        method = hist_eq_config.get("method", "clahe")
        clahe_config = hist_eq_config.get("clahe", {})
        clip_limit = clahe_config.get("clip_limit", 2.0)
        tile_grid = clahe_config.get("tile_grid_size", [8, 8])
        tile_grid_size = tuple(tile_grid) if isinstance(tile_grid, list) else tile_grid
    else:
        method = "clahe"
        clip_limit = 2.0
        tile_grid_size = (8, 8)
    
    # グローバルヒストグラム均等化
    hist_eq_global = HistogramEqualization(method="global")
    result_global = hist_eq_global(image)
    
    # CLAHE（コントラスト制限付き適応ヒストグラム均等化）
    hist_eq_clahe = HistogramEqualization(
        method=method,
    )
    result_clahe = hist_eq_clahe(image)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(result_global)
    axes[1].set_title("Global Histogram Equalization")
    axes[1].axis("off")
    
    axes[2].imshow(result_clahe)
    axes[2].set_title("CLAHE")
    axes[2].axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_histogram_eq.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存しました: {output_path}")
    plt.show()


def demo_gamma_correction(image_path: str, config: dict = None):
    """ガンマ補正のデモ"""
    print("\n" + "=" * 60)
    print("ガンマ補正のデモ")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    
    # 設定からガンマ値を取得（なければデフォルト値を使用）
    if config:
        gamma_config = config.get("gamma_correction", {})
        default_gamma = gamma_config.get("gamma", 1.2)
    else:
        default_gamma = 1.2
    
    # 異なるガンマ値で補正（設定値も含める）
    gamma_values = [0.5, 1.0, default_gamma, 2.0]
    if default_gamma not in gamma_values:
        gamma_values = [0.5, 1.0, default_gamma, 2.0]
    
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(20, 5))
    
    for i, gamma in enumerate(gamma_values):
        gamma_corr = GammaCorrection(gamma=gamma)
        result = gamma_corr(image)
        
        axes[i].imshow(result)
        axes[i].set_title(f"Gamma = {gamma}")
        axes[i].axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_gamma.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存しました: {output_path}")
    plt.show()


def demo_patching(image_path: str, config: dict = None):
    """パッチ化のデモ"""
    print("\n" + "=" * 60)
    print("パッチ化のデモ")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    
    print(f"元画像サイズ: {image.shape}")
    
    # 設定からパラメータを取得
    if config:
        patching_config = config.get("patching", {})
        patch_size = patching_config.get("patch_size", [14, 14])
        stride = patching_config.get("stride", None)
        padding = patching_config.get("padding", True)
        
        # リストをタプルに変換
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        if stride is not None and isinstance(stride, list):
            stride = tuple(stride)
        
        # 画像サイズに合わせて調整
        patch_h = min(patch_size[0], image.shape[0] // 2)
        patch_w = min(patch_size[1], image.shape[1] // 2)
        
        if stride is None:
            stride_h = max(patch_h // 2, 1)
            stride_w = max(patch_w // 2, 1)
        else:
            stride_h = stride[0]
            stride_w = stride[1]
    else:
        # デフォルト値（MNIST用）
        patch_h = min(14, image.shape[0] // 2)
        patch_w = min(14, image.shape[1] // 2)
        stride_h = max(7, patch_h // 2)
        stride_w = max(7, patch_w // 2)
        padding = True
    
    print(f"パッチサイズ: ({patch_h}, {patch_w})")
    print(f"ストライド: ({stride_h}, {stride_w})")
    print(f"パディング: {padding}")
    
    # パッチ化
    patching = Patching(
        patch_size=(patch_h, patch_w),
        stride=(stride_h, stride_w),
        padding=padding
    )
    patches = patching(image)
    
    if len(patches) == 0:
        print("\n警告: パッチが生成されませんでした")
        print("画像サイズがパッチサイズより小さい可能性があります")
        return
    
    print(f"パッチ数: {len(patches)}")
    print(f"各パッチサイズ: {patches[0].shape}")
    
    # 最初の16パッチを表示
    num_display = min(16, len(patches))
    rows = 4
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i in range(num_display):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(patches[i])
        axes[row, col].set_title(f"Patch {i}")
        axes[row, col].axis("off")
    
    # 余ったサブプロットを非表示
    for i in range(num_display, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_patching.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存しました: {output_path}")
    plt.show()


def demo_pipeline(image_path: str, preprocessing_config: dict = None, image_config: dict = None):
    """前処理パイプラインのデモ"""
    print("\n" + "=" * 60)
    print("前処理パイプラインのデモ")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    print(f"元画像サイズ: {image.shape}")
    
    # 設定からパイプラインを構築
    if preprocessing_config:
        pipeline = create_preprocessing_pipeline(
            preprocessing_config,
            image_config=image_config
        )
        if pipeline is None:
            print("警告: 有効な前処理が設定されていません")
            print("auguments.yamlのpreprocessingセクションで前処理を有効にしてください")
            return
    else:
        # デフォルトのパイプライン（リサイズ含む）
        transforms = [
            HistogramEqualization(method="clahe"),
            GammaCorrection(gamma=1.2)
        ]
        # image_configがあればリサイズも追加（Noneや空でないことを確認）
        if image_config:
            image_size = image_config.get("size")
            if image_size is not None and image_size:
                if isinstance(image_size, list):
                    image_size = tuple(image_size)
                # 有効なサイズであることを確認
                if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                    transforms.append(Resize(size=image_size))
        pipeline = PreprocessingPipeline(transforms)
    
    print(pipeline)
    
    # パイプラインを適用（常にリストが返る）
    result_list = pipeline(image)
    
    print(f"\n出力画像数: {len(result_list)}")
    print(f"出力画像サイズ: {result_list[0].shape}")
    if pipeline.has_patching:
        print("パッチ化が有効です。最初の数パッチを表示します。")
    
    # パッチ化なしの場合は1枚、ありの場合は最大16枚表示
    num_display = min(16, len(result_list))
    
    if num_display == 1:
        # パッチ化なし: 元画像と処理後画像を並べて表示
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(result_list[0])
        axes[1].set_title("After Pipeline")
        axes[1].axis("off")
    else:
        # パッチ化あり: 元画像とパッチを表示
        rows = 5
        cols = 4
        fig = plt.figure(figsize=(16, 20))
        
        # 元画像を最初に表示
        ax = plt.subplot(rows, cols, 1)
        ax.imshow(image)
        ax.set_title("Original Image")
        ax.axis("off")
        
        # パッチを表示（最大16枚）
        for i in range(num_display):
            ax = plt.subplot(rows, cols, i + 2)
            ax.imshow(result_list[i])
            ax.set_title(f"Patch {i}")
            ax.axis("off")
        
        # 余ったサブプロットを非表示
        for i in range(num_display + 1, rows * cols):
            ax = plt.subplot(rows, cols, i + 1)
            ax.axis("off")
    
    plt.tight_layout()
    output_path = project_root / "workspace" / "demo_pipeline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存しました: {output_path}")
    plt.show()


def get_sample_image_from_theme(theme_id: int) -> str:
    """
    Djangoテーマからランダムに画像を1枚取得
    
    Args:
        theme_id: テーマID
        
    Returns:
        画像ファイルパス
    """
    # テーマの存在確認
    theme = get_theme(theme_id=theme_id)
    if theme is None:
        raise ValueError(f"テーマID {theme_id} が見つかりません")
    
    print(f"\nテーマ情報:")
    print(f"  ID: {theme.id}")
    print(f"  名前: {theme.name}")
    print(f"  説明: {theme.description or '(なし)'}")
    
    # テーマの全画像を取得
    train_data_list = get_traindata_by_theme(theme_id=theme_id)
    
    if not train_data_list:
        raise ValueError(f"テーマID {theme_id} に画像が登録されていません")
    
    print(f"  登録画像数: {len(train_data_list)}枚")
    
    # ランダムに1枚選択
    sample_data = random.choice(train_data_list)
    image_path = sample_data.image.path
    
    print(f"\nサンプル画像:")
    print(f"  パス: {image_path}")
    print(f"  ラベル: {sample_data.label.label_name}")
    
    return image_path


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description="前処理の動作確認デモ（Django統合版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme-id",
        type=int,
        required=True,
        help="テーマID"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("前処理デモスクリプト（Django統合版）")
    print("=" * 60)
    
    # auguments.yamlから設定を読み込み
    augments_file = Path("auguments.yaml")
    preprocessing_config = None
    image_config = None
    if augments_file.exists():
        print(f"\n✓ {augments_file} から設定を読み込み中...")
        with open(augments_file, "r") as f:
            all_config = yaml.safe_load(f)
            preprocessing_config = all_config.get("preprocessing", {})
            image_config = all_config.get("image", {})
            if preprocessing_config:
                print("  前処理設定を読み込みました")
            else:
                print("  警告: preprocessingセクションが見つかりません")
            if image_config:
                print(f"  画像設定を読み込みました (size={image_config.get('size')})")
    else:
        print(f"\n警告: {augments_file} が見つかりません")
        print("  デフォルト設定を使用します")
    
    try:
        # Djangoテーマから画像を取得
        image_path = get_sample_image_from_theme(args.theme_id)
        
        # 各デモを実行（設定を渡す）
        demo_histogram_equalization(image_path, preprocessing_config)
        demo_gamma_correction(image_path, preprocessing_config)
        demo_patching(image_path, preprocessing_config)
        demo_pipeline(image_path, preprocessing_config, image_config)
        
        print("\n" + "=" * 60)
        print("完了")
        print("=" * 60)
        print("\n生成された画像:")
        print("  - workspace/demo_histogram_eq.png")
        print("  - workspace/demo_gamma.png")
        print("  - workspace/demo_patching.png")
        print("  - workspace/demo_pipeline.png")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)


