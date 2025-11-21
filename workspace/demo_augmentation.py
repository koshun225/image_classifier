"""
オーグメンテーションの動作確認デモ

データオーグメンテーションの各変換を可視化して確認します。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.augmentation import AugmentationBuilder, get_transforms
from src.data.preprocessing import load_image


def demo_train_augmentations(image_path: str, num_samples: int = 8):
    """学習用オーグメンテーションのデモ"""
    print("\n" + "=" * 60)
    print("学習用オーグメンテーションのデモ")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    
    # 学習用変換を取得
    transform = get_transforms("auguments.yaml", split="train")
    
    print(f"使用する変換:\n{transform}\n")
    
    # 同じ画像に複数回変換を適用
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        # 変換を適用
        try:
            # albumentationsの場合
            if hasattr(transform, '__class__') and 'albumentations' in str(type(transform)):
                transformed = transform(image=image)
                aug_image = transformed['image']
                # テンソルをnumpy配列に変換して表示
                if len(aug_image.shape) == 3:
                    # (C, H, W) -> (H, W, C)
                    aug_image = aug_image.permute(1, 2, 0).numpy()
            # torchvisionの場合
            else:
                pil_image = Image.fromarray(image)
                aug_image = transform(pil_image)
                # テンソルをnumpy配列に変換
                aug_image = aug_image.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(aug_image)
            axes[row, col].set_title(f"Sample {i+1}")
            axes[row, col].axis("off")
        except Exception as e:
            print(f"エラー (Sample {i+1}): {e}")
            axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig("workspace/demo_train_augmentation.png", dpi=150, bbox_inches='tight')
    print("✓ 保存しました: workspace/demo_train_augmentation.png")
    plt.show()


def demo_val_test_transform(image_path: str):
    """検証/テスト用変換のデモ"""
    print("\n" + "=" * 60)
    print("検証/テスト用変換のデモ")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    
    # 変換を取得
    val_transform = get_transforms("auguments.yaml", split="val")
    test_transform = get_transforms("auguments.yaml", split="test")
    
    print(f"検証用変換:\n{val_transform}\n")
    print(f"テスト用変換:\n{test_transform}\n")
    
    # 変換を適用
    try:
        # albumentationsの場合
        if hasattr(val_transform, '__class__') and 'albumentations' in str(type(val_transform)):
            val_result = val_transform(image=image)
            val_image = val_result['image'].permute(1, 2, 0).numpy()
            
            test_result = test_transform(image=image)
            test_image = test_result['image'].permute(1, 2, 0).numpy()
        # torchvisionの場合
        else:
            pil_image = Image.fromarray(image)
            val_image = val_transform(pil_image).permute(1, 2, 0).numpy()
            test_image = test_transform(pil_image).permute(1, 2, 0).numpy()
        
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(val_image)
        axes[1].set_title("Val Transform\n(Resize + CenterCrop + Normalize)")
        axes[1].axis("off")
        
        axes[2].imshow(test_image)
        axes[2].set_title("Test Transform\n(Resize + CenterCrop + Normalize)")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig("workspace/demo_val_test_transform.png", dpi=150, bbox_inches='tight')
        print("✓ 保存しました: workspace/demo_val_test_transform.png")
        plt.show()
        
    except Exception as e:
        print(f"エラー: {e}")


def compare_libraries(image_path: str):
    """albumentations と torchvision の比較"""
    print("\n" + "=" * 60)
    print("albumentations vs torchvision")
    print("=" * 60)
    
    # 画像を読み込み
    image = load_image(image_path)
    
    # albumentations
    builder_albu = AugmentationBuilder("auguments.yaml")
    if builder_albu.library == "albumentations":
        print("使用ライブラリ: albumentations ✓")
        transform_albu = builder_albu.build_train_transform()
        
        result_albu = transform_albu(image=image)
        image_albu = result_albu['image'].permute(1, 2, 0).numpy()
    else:
        print("使用ライブラリ: torchvision (albumentations未インストール)")
        transform_albu = builder_albu.build_train_transform()
        
        pil_image = Image.fromarray(image)
        image_albu = transform_albu(pil_image).permute(1, 2, 0).numpy()
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(image_albu)
    axes[1].set_title(f"After Augmentation\n(Library: {builder_albu.library})")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("workspace/demo_library_comparison.png", dpi=150, bbox_inches='tight')
    print("✓ 保存しました: workspace/demo_library_comparison.png")
    plt.show()


def main():
    """メイン処理"""
    print("=" * 60)
    print("オーグメンテーションデモスクリプト")
    print("=" * 60)
    
    # auguments.yamlの存在確認
    if not Path("auguments.yaml").exists():
        print("\nエラー: auguments.yaml が見つかりません")
        return
    
    print("\n✓ auguments.yaml から設定を読み込みます")
    
    # 本番データのパスを探す
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("\nエラー: data/ が見つかりません")
        print("本番データを data/ ディレクトリに配置してください")
        print("\n代替: テスト用データを使う場合は data_for_test/ を使用します")
        data_dir = Path("data_for_test")
        if not data_dir.exists():
            print("エラー: data_for_test/ も見つかりません")
            print("先に workspace/mnist_data_prepare.py を実行してください")
            return
    
    print(f"使用するデータディレクトリ: {data_dir}")
    
    # 最初に見つかった画像を使用
    image_path = None
    for version_dir in sorted(data_dir.iterdir()):
        if version_dir.is_dir():
            for class_dir in sorted(version_dir.iterdir()):
                if class_dir.is_dir():
                    images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
                    if images:
                        image_path = str(images[0])
                        break
            if image_path:
                break
    
    if not image_path:
        print(f"\nエラー: {data_dir} 内に画像が見つかりません")
        return
    
    print(f"テスト画像: {image_path}")
    
    # 各デモを実行
    demo_train_augmentations(image_path, num_samples=8)
    demo_val_test_transform(image_path)
    compare_libraries(image_path)
    
    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)
    print("\n生成された画像:")
    print("  - workspace/demo_train_augmentation.png")
    print("  - workspace/demo_val_test_transform.png")
    print("  - workspace/demo_library_comparison.png")


if __name__ == "__main__":
    main()


