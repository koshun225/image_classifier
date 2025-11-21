#!/usr/bin/env python3
"""
MNISTデータを準備して、1,000枚を3つのバージョンに分割してdata_for_test/ver1, ver2, ver3に配置するスクリプト
テスト専用のデータセットです。
"""

import shutil
from pathlib import Path
import numpy as np
from torchvision import datasets
from PIL import Image


def download_mnist():
    """MNISTデータをダウンロード"""
    print("MNISTデータをダウンロード中...")
    # 一時ディレクトリにダウンロード
    temp_dir = Path("workspace/temp_mnist")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # MNISTデータセットをダウンロード（train=Trueで訓練データを取得）
    mnist_dataset = datasets.MNIST(
        root=str(temp_dir),
        train=True,
        download=True,
        transform=None  # 生データを取得
    )
    
    return mnist_dataset


def split_mnist_by_version(mnist_dataset, num_versions=3, total_samples=1000):
    """
    MNISTデータを指定枚数に制限して3つのバージョンに分割
    
    Args:
        mnist_dataset: MNISTデータセット
        num_versions: 分割するバージョン数
        total_samples: 使用する総サンプル数（デフォルト: 1000枚）
    
    Returns:
        versions: 各バージョンのデータ（クラス別に整理）
    """
    print(f"MNISTデータを{total_samples}枚に制限して{num_versions}つのバージョンに分割中...")
    
    # クラス別にデータを整理
    class_data = {i: [] for i in range(10)}  # MNISTは0-9の10クラス
    
    for idx in range(len(mnist_dataset)):
        image, label = mnist_dataset[idx]
        class_data[label].append((image, label, idx))
    
    # 各クラスから均等にサンプリング（合計1000枚 = 各クラス100枚）
    samples_per_class = total_samples // 10
    np.random.seed(42)  # 再現性のためシードを固定
    
    sampled_class_data = {}
    for class_label in range(10):
        class_samples = class_data[class_label]
        np.random.shuffle(class_samples)
        # 各クラスから指定枚数をサンプリング
        sampled_class_data[class_label] = class_samples[:samples_per_class]
        print(f"  クラス{class_label}: {len(sampled_class_data[class_label])}枚をサンプリング")
    
    # 各クラスのサンプルを3つのバージョンに分割
    versions = {f"ver{i+1}": {j: [] for j in range(10)} for i in range(num_versions)}
    
    for class_label in range(10):
        class_samples = sampled_class_data[class_label]
        
        # 各クラスのデータを3つに分割
        total_class_samples = len(class_samples)
        samples_per_version = total_class_samples // num_versions
        
        for version_idx in range(num_versions):
            start_idx = version_idx * samples_per_version
            if version_idx == num_versions - 1:
                # 最後のバージョンには残りすべてを含める
                end_idx = total_class_samples
            else:
                end_idx = (version_idx + 1) * samples_per_version
            
            version_name = f"ver{version_idx + 1}"
            versions[version_name][class_label] = class_samples[start_idx:end_idx]
    
    return versions


def save_images_to_class_folders(versions, base_dir="data"):
    """
    各バージョンのデータをクラス別フォルダ構造で保存
    
    Args:
        versions: 各バージョンのデータ（クラス別に整理）
        base_dir: 保存先のベースディレクトリ
    """
    print("画像をクラス別フォルダ構造で保存中...")
    
    base_path = Path(base_dir)
    
    for version_name, class_data in versions.items():
        version_path = base_path / version_name
        version_path.mkdir(parents=True, exist_ok=True)
        
        for class_label in range(10):
            class_folder = version_path / f"{class_label}"
            class_folder.mkdir(parents=True, exist_ok=True)
            
            for idx, (image, label, original_idx) in enumerate(class_data[class_label]):
                # 画像を保存
                image_filename = f"{original_idx:05d}.png"
                image_path = class_folder / image_filename
                image.save(image_path)
        
        print(f"  {version_name}: {sum(len(class_data[c]) for c in range(10))}枚の画像を保存しました")
        
        # 各クラスの画像数を表示
        for class_label in range(10):
            num_images = len(class_data[class_label])
            print(f"    class{class_label}: {num_images}枚")


def cleanup_temp_files():
    """一時ファイルを削除"""
    temp_dir = Path("workspace/temp_mnist")
    if temp_dir.exists():
        print("一時ファイルを削除中...")
        shutil.rmtree(temp_dir)


def main():
    """メイン処理"""
    print("=" * 60)
    print("MNISTテストデータ準備スクリプト")
    print("=" * 60)
    
    # テスト用データディレクトリの確認・作成
    data_dir = Path("data_for_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"テスト用データディレクトリ: {data_dir}")
    
    # MNISTデータをダウンロード
    mnist_dataset = download_mnist()
    print(f"MNISTデータセット: {len(mnist_dataset)}枚の画像")
    
    # データを1,000枚に制限して3つのバージョンに分割
    versions = split_mnist_by_version(mnist_dataset, num_versions=3, total_samples=1000)
    
    # 各バージョンの統計情報を表示
    print("\n各バージョンのデータ数:")
    for version_name, class_data in versions.items():
        total = sum(len(class_data[c]) for c in range(10))
        print(f"  {version_name}: {total}枚")
        for class_label in range(10):
            print(f"    class{class_label}: {len(class_data[class_label])}枚")
    
    # 画像をクラス別フォルダ構造で保存
    print("\n画像を保存中...")
    save_images_to_class_folders(versions, base_dir="data_for_test")
    
    # 一時ファイルを削除
    cleanup_temp_files()
    
    print("\n" + "=" * 60)
    print("完了しました！")
    print("=" * 60)
    print("\nテストデータの配置:")
    print("  data_for_test/ver1/class0/, class1/, ..., class9/")
    print("  data_for_test/ver2/class0/, class1/, ..., class9/")
    print("  data_for_test/ver3/class0/, class1/, ..., class9/")
    print("\n注意: このデータはテスト専用です。")
    print("本番用データは data/ ディレクトリに配置してください。")


if __name__ == "__main__":
    main()

