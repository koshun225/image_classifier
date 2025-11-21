"""
pytest設定とfixture
Djangoデータベースベースのテスト用fixture
"""

import pytest
from pathlib import Path
import yaml
import os
import sys
import shutil
import tempfile
from PIL import Image
import numpy as np


@pytest.fixture(scope="session")
def project_root():
    """プロジェクトルートのパスを返す"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def params(project_root):
    """params.yamlを読み込む"""
    params_file = project_root / "params.yaml"
    with open(params_file, "r") as f:
        schema = yaml.safe_load(f)
    sys.path.insert(0, str(project_root))
    from src.utils.params_schema import materialize_params
    return materialize_params(schema or {})


@pytest.fixture(scope="session")
def config(project_root):
    """config.yamlを読み込む"""
    config_file = project_root / "config.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    else:
        # デフォルト設定
        return {"data": {"test_dir": "data_for_test"}}


@pytest.fixture(scope="session")
def setup_django_env(project_root):
    """Django環境をセットアップ"""
    sys.path.insert(0, str(project_root / 'src' / 'web'))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    
    import django
    django.setup()
    
    # マイグレーションを実行してテーブルを作成
    from django.core.management import call_command
    from django.db import connection
    
    # テーブルを作成
    call_command('migrate', '--run-syncdb', verbosity=0)
    
    return django


@pytest.fixture(scope="function")
def test_theme(setup_django_env):
    """テスト用のテーマを作成"""
    from data_management.crud import create_theme, create_label
    
    # テーマを作成
    theme = create_theme(name="MNIST Test", description="Test theme for MNIST")
    
    # ラベルを作成（0-9）
    labels = []
    for i in range(10):
        label = create_label(theme_id=theme.id, label_name=str(i))
        labels.append(label)
    
    yield theme, labels
    
    # クリーンアップ
    theme.delete()


@pytest.fixture(scope="function")
def create_dummy_mnist_images():
    """ダミーのMNIST画像を作成するヘルパー関数"""
    def _create_images(num_images_per_class=10):
        """
        ダミーMNIST画像を作成
        
        Args:
            num_images_per_class: クラスあたりの画像数
        
        Returns:
            images: {class_id: [image_paths]}
        """
        temp_dir = Path(tempfile.mkdtemp())
        images = {}
        
        for class_id in range(10):
            class_dir = temp_dir / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)
            
            class_images = []
            for img_idx in range(num_images_per_class):
                # ダミー画像を作成（28x28のグレースケール）
                img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
                # クラスに応じて特徴を追加（テスト用）
                img_array[10:18, 10:18] = class_id * 25
                
                img = Image.fromarray(img_array, mode='L').convert('RGB')
                img_path = class_dir / f"{img_idx:05d}.png"
                img.save(img_path)
                class_images.append(str(img_path))
            
            images[class_id] = class_images
        
        yield temp_dir, images
        
        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return _create_images


@pytest.fixture(scope="function")
def mnist_test_data(test_theme, create_dummy_mnist_images, setup_django_env):
    """MNISTテストデータをデータベースに登録"""
    from data_management.crud import create_traindata, get_labels_by_theme
    from django.core.files import File
    
    theme, _ = test_theme
    labels = get_labels_by_theme(theme_id=theme.id)
    
    # ダミー画像を作成
    with create_dummy_mnist_images(num_images_per_class=10) as (temp_dir, images):
        for class_id, image_paths in images.items():
            label = labels[class_id]
            
            for img_path in image_paths:
                # データベースに登録
                with open(img_path, 'rb') as f:
                    django_file = File(f, name=Path(img_path).name)
                    create_traindata(
                        theme_id=theme.id,
                        image=django_file,
                        label_id=label.id,
                        labeled_by="test_user"
                    )
    
    return theme, labels


@pytest.fixture
def temp_splits_dir(tmp_path):
    """テスト用の一時分割ディレクトリを作成（後方互換性のため）"""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    return splits_dir
