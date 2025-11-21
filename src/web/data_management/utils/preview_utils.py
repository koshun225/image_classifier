"""
augmentation/preprocessingプレビュー画像生成ユーティリティ
"""
import sys
import os
from pathlib import Path
import numpy as np
import base64
import io
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data.augmentation import AugmentationBuilder, get_transforms
    from src.data.preprocessing import load_image, create_preprocessing_pipeline, PreprocessingPipeline
except ImportError:
    # Django環境外でも動作するように
    pass

logger = logging.getLogger(__name__)


def get_sample_images(theme_id: int, num_images: int = 5) -> List[str]:
    """
    テーマからサンプル画像のパスを取得
    
    Args:
        theme_id: テーマID
        num_images: 取得する画像数
    
    Returns:
        画像パスのリスト
    """
    # Django環境のセットアップ
    sys.path.insert(0, str(project_root / 'src' / 'web'))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    
    try:
        import django
        django.setup()
        from data_management.crud import get_traindata_by_theme
        from data_management.models import TrainData
        
        # テーマの画像を取得
        train_data_list = TrainData.objects.filter(theme_id=theme_id, split__in=['train', 'valid', 'test'])[:num_images]
        
        image_paths = []
        for train_data in train_data_list:
            image_path = train_data.image.path
            if os.path.exists(image_path):
                image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        logger.error(f"サンプル画像取得エラー: {e}")
        return []


def image_to_base64(image_array: np.ndarray) -> str:
    """
    numpy配列の画像をBase64エンコード
    
    Args:
        image_array: (H, W, C) のnumpy配列
    
    Returns:
        Base64エンコードされた文字列（data URI形式）
    """
    # 値を0-255の範囲にクリップ
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    # PIL Imageに変換
    if len(image_array.shape) == 2:
        # グレースケール
        pil_image = Image.fromarray(image_array, mode='L')
    else:
        # RGB
        pil_image = Image.fromarray(image_array, mode='RGB')
    
    # Base64エンコード
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def generate_preprocessing_preview(
    theme_id: int,
    auguments_config: Dict[str, Any],
    num_images: int = 5
) -> Dict[str, Any]:
    """
    前処理プレビュー画像を生成
    
    Args:
        theme_id: テーマID
        auguments_config: auguments.yamlの設定（辞書）
        num_images: 生成する画像数
    
    Returns:
        {
            'original': [base64_images],
            'preprocessed': [base64_images]
        }
    """
    try:
        # サンプル画像を取得
        image_paths = get_sample_images(theme_id, num_images)
        if not image_paths:
            return {'original': [], 'preprocessed': []}
        
        # 前処理パイプラインを構築
        preprocessing_config = auguments_config.get('preprocessing', {})
        image_config = auguments_config.get('image', {})
        
        pipeline = create_preprocessing_pipeline(
            preprocessing_config,
            image_config=image_config
        )
        
        original_images = []
        preprocessed_images = []
        
        for image_path in image_paths:
            # 元画像を読み込み
            original_image = load_image(image_path)
            original_images.append(image_to_base64(original_image))
            
            # 前処理を適用
            if pipeline is not None:
                result_list = pipeline(original_image)
                # 最初の結果を使用（パッチ化の場合は最初のパッチ）
                preprocessed_image = result_list[0] if result_list else original_image
            else:
                preprocessed_image = original_image
            
            preprocessed_images.append(image_to_base64(preprocessed_image))
        
        return {
            'original': original_images,
            'preprocessed': preprocessed_images
        }
    except Exception as e:
        logger.error(f"前処理プレビュー生成エラー: {e}")
        return {'original': [], 'preprocessed': [], 'error': str(e)}


def generate_augmentation_preview(
    theme_id: int,
    auguments_config: Dict[str, Any],
    num_images: int = 5,
    num_samples: int = 5
) -> Dict[str, Any]:
    """
    オーグメンテーションプレビュー画像を生成
    
    Args:
        theme_id: テーマID
        auguments_config: auguments.yamlの設定（辞書）
        num_images: 使用する元画像数
        num_samples: 各画像から生成するサンプル数
    
    Returns:
        {
            'original': [base64_images],
            'augmented': [[base64_images_per_image], ...]
        }
    """
    try:
        # 一時的にauguments.yamlを保存
        temp_config_path = project_root / "auguments_temp.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(auguments_config, f, default_flow_style=False, allow_unicode=True)
        
        # サンプル画像を取得
        image_paths = get_sample_images(theme_id, num_images)
        if not image_paths:
            return {'original': [], 'augmented': []}
        
        # 学習用変換を取得
        transform = get_transforms(str(temp_config_path), split="train")
        
        original_images = []
        augmented_images_list = []
        
        for image_path in image_paths:
            # 元画像を読み込み
            original_image = load_image(image_path)
            original_images.append(image_to_base64(original_image))
            
            # 複数回変換を適用
            augmented_samples = []
            for _ in range(num_samples):
                try:
                    # albumentationsの場合
                    if hasattr(transform, '__class__') and 'albumentations' in str(type(transform)):
                        transformed = transform(image=original_image)
                        aug_image = transformed['image']
                        # テンソルをnumpy配列に変換
                        if hasattr(aug_image, 'permute'):
                            # (C, H, W) -> (H, W, C)
                            aug_image = aug_image.permute(1, 2, 0).numpy()
                        else:
                            aug_image = np.array(aug_image)
                    # torchvisionの場合
                    else:
                        pil_image = Image.fromarray(original_image)
                        aug_image = transform(pil_image)
                        # テンソルをnumpy配列に変換
                        if hasattr(aug_image, 'permute'):
                            aug_image = aug_image.permute(1, 2, 0).numpy()
                        else:
                            aug_image = np.array(aug_image)
                    
                    # 正規化を元に戻す（必要に応じて）
                    if aug_image.min() < 0 or aug_image.max() <= 1.0:
                        # 正規化されている可能性がある
                        aug_image = (aug_image - aug_image.min()) / (aug_image.max() - aug_image.min() + 1e-8)
                        aug_image = (aug_image * 255).astype(np.uint8)
                    
                    augmented_samples.append(image_to_base64(aug_image))
                except Exception as e:
                    logger.warning(f"オーグメンテーション適用エラー: {e}")
                    # エラー時は元画像を使用
                    augmented_samples.append(image_to_base64(original_image))
            
            augmented_images_list.append(augmented_samples)
        
        # 一時ファイルを削除
        if temp_config_path.exists():
            temp_config_path.unlink()
        
        return {
            'original': original_images,
            'augmented': augmented_images_list
        }
    except Exception as e:
        logger.error(f"オーグメンテーションプレビュー生成エラー: {e}")
        return {'original': [], 'augmented': [], 'error': str(e)}

