"""
MLflow PythonModel継承クラス

カスタムPythonModelとしてモデルを登録します。
前処理と推論を統合し、推論時に前処理が自動的に適用されます。
"""

import mlflow.pyfunc
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Any
import logging
import yaml
from pathlib import Path

from src.data.preprocessing import create_preprocessing_pipeline
from src.data.augmentation import get_transforms

logger = logging.getLogger(__name__)


class ClassificationPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    画像分類用のカスタムPythonModel
    
    前処理と推論を統合し、MLflow Models API経由で推論を実行します。
    """
    
    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        """
        モデルと設定のロード
        
        Args:
            context: MLflowのコンテキスト（artifacts pathなど）
        """
        # PyTorchモデルのロード
        model_path = context.artifacts["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        logger.info(f"モデルをロードしました: {model_path}")
        logger.info(f"デバイス: {self.device}")
        
        # 前処理設定のロード
        preprocessing_config_path = context.artifacts.get("preprocessing_config")
        augments_config_path = context.artifacts.get("augments_config")
        
        self.preprocessing_pipeline = None
        self.transform = None
        
        if preprocessing_config_path:
            try:
                with open(preprocessing_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    preprocessing_config = config.get("preprocessing", {})
                    image_config = config.get("image", {})
                    
                    if preprocessing_config:
                        self.preprocessing_pipeline = create_preprocessing_pipeline(
                            preprocessing_config,
                            image_config=image_config
                        )
                        logger.info("前処理パイプラインをロードしました")
            except Exception as e:
                logger.warning(f"前処理設定のロードに失敗: {e}")
        
        if augments_config_path:
            try:
                # テスト時の変換を使用（オーグメンテーションなし）
                self.transform = get_transforms(augments_config_path, split="test")
                logger.info("変換をロードしました")
            except Exception as e:
                logger.warning(f"変換設定のロードに失敗: {e}")
        
        # クラス名のロード（オプション）
        class_names_path = context.artifacts.get("class_names")
        if class_names_path:
            try:
                with open(class_names_path, "r") as f:
                    self.class_names = [line.strip() for line in f]
                logger.info(f"クラス名をロードしました: {len(self.class_names)}クラス")
            except Exception as e:
                logger.warning(f"クラス名のロードに失敗: {e}")
                self.class_names = None
        else:
            self.class_names = None
    
    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Union[np.ndarray, List]
    ) -> np.ndarray:
        """
        推論を実行
        
        Args:
            context: MLflowのコンテキスト
            model_input: 入力データ
                - numpy配列の場合: [batch_size, height, width, channels]
                - リストの場合: 画像パスのリスト
        
        Returns:
            予測結果 [batch_size, num_classes]
        """
        # 入力の前処理
        images = []
        
        if isinstance(model_input, list):
            # 画像パスのリスト
            for img_path in model_input:
                image = self._load_and_preprocess_image(img_path)
                images.append(image)
        elif isinstance(model_input, np.ndarray):
            # NumPy配列（バッチ）
            for i in range(model_input.shape[0]):
                img = model_input[i]
                image = self._preprocess_image(img)
                images.append(image)
        else:
            raise ValueError(
                f"サポートされていない入力タイプ: {type(model_input)}. "
                "numpy配列またはリストが必要です。"
            )
        
        # バッチテンソルの作成
        if len(images) == 0:
            raise ValueError("入力画像が空です")
        
        batch_tensor = torch.stack(images).to(self.device)
        
        # 推論
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            predictions = outputs.cpu().numpy()
        
        return predictions
    
    def _load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        """
        画像をロードして前処理
        
        Args:
            img_path: 画像のパス
        
        Returns:
            前処理済みのテンソル
        """
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        return self._preprocess_image(image_np)
    
    def _preprocess_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        画像を前処理
        
        Args:
            image_np: NumPy配列の画像 [height, width, channels]
        
        Returns:
            前処理済みのテンソル [channels, height, width]
        """
        # 前処理パイプラインを適用
        if self.preprocessing_pipeline is not None:
            image_list = self.preprocessing_pipeline(image_np)
            image_np = image_list[0]  # 最初の画像を使用
        
        # 変換を適用
        if self.transform is not None:
            # albumentationsの場合
            if hasattr(self.transform, '__class__') and 'albumentations' in str(type(self.transform)):
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            # torchvisionの場合
            else:
                image_pil = Image.fromarray(image_np)
                image_tensor = self.transform(image_pil)
        else:
            # transformがない場合はテンソルに変換
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        return image_tensor


def log_model(
    model: torch.nn.Module,
    artifact_path: str,
    preprocessing_config_path: str = None,
    augments_config_path: str = None,
    class_names_path: str = None,
    registered_model_name: str = None,
    **kwargs
):
    """
    カスタムPythonModelとしてモデルをMLflowに登録
    
    Args:
        model: PyTorchモデル
        artifact_path: MLflow内のartifact path
        preprocessing_config_path: 前処理設定ファイルのパス
        augments_config_path: オーグメンテーション設定ファイルのパス
        class_names_path: クラス名ファイルのパス
        registered_model_name: 登録するモデル名
        **kwargs: その他のmlflow.pyfunc.log_modelのパラメータ
    """
    # Artifactsの準備
    artifacts = {
        "model": "model.pth"
    }
    
    # モデルの保存
    model_save_path = Path("model.pth")
    torch.save(model, model_save_path)
    
    if preprocessing_config_path:
        artifacts["preprocessing_config"] = preprocessing_config_path
    
    if augments_config_path:
        artifacts["augments_config"] = augments_config_path
    
    if class_names_path:
        artifacts["class_names"] = class_names_path
    
    # カスタムPythonModelとしてログ
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=ClassificationPyFuncModel(),
        artifacts=artifacts,
        registered_model_name=registered_model_name,
        **kwargs
    )
    
    # 一時ファイルを削除
    if model_save_path.exists():
        model_save_path.unlink()
    
    logger.info(f"モデルをMLflowに登録しました: {artifact_path}")


if __name__ == "__main__":
    # テスト用のダミー実装
    print("MLflow PythonModelモジュール")
    print("このモジュールは、カスタムPythonModelとしてモデルをMLflowに登録します。")
