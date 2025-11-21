"""
YAMLファイル読み込み・編集ユーティリティ
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import sys

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    # src/web/data_management/utils/yaml_utils.py から
    # プロジェクトルート（classification_with_mlops）を取得
    current_file = Path(__file__).resolve()
    # src/web/data_management/utils -> src/web -> src -> プロジェクトルート
    project_root = current_file.parent.parent.parent.parent.parent
    return project_root


def load_yaml_file(filename: str) -> Dict[str, Any]:
    """
    YAMLファイルを読み込む
    
    Args:
        filename: ファイル名（例: 'params.yaml'）
    
    Returns:
        YAMLの内容（辞書）
    """
    project_root = get_project_root()
    file_path = project_root / filename
    
    if not file_path.exists():
        logger.warning(f"ファイルが見つかりません: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        logger.error(f"YAMLファイルの読み込みエラー: {e}")
        raise
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        raise


def save_yaml_file(filename: str, data: Dict[str, Any]) -> bool:
    """
    YAMLファイルを保存する
    
    Args:
        filename: ファイル名（例: 'params.yaml'）
        data: 保存するデータ（辞書）
    
    Returns:
        成功した場合True
    """
    project_root = get_project_root()
    file_path = project_root / filename
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"YAMLファイルを保存しました: {file_path}")
        return True
    except Exception as e:
        logger.error(f"YAMLファイルの保存エラー: {e}")
        raise


def validate_yaml_content(content: str) -> Tuple[bool, Optional[str]]:
    """
    YAML文字列のバリデーション
    
    Args:
        content: YAML文字列
    
    Returns:
        (is_valid, error_message)
    """
    try:
        yaml.safe_load(content)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)


def get_yaml_file_content(filename: str) -> str:
    """
    YAMLファイルの内容を文字列として取得
    
    Args:
        filename: ファイル名
    
    Returns:
        YAMLファイルの内容（文字列）
    """
    project_root = get_project_root()
    file_path = project_root / filename
    
    if not file_path.exists():
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        return ""


PARAM_META_KEYS = {'value', 'type', 'low', 'high', 'choices', 'log', 'step'}


def _is_param_node(node: Dict[str, Any]) -> bool:
    return any(key in PARAM_META_KEYS for key in node.keys())


def normalize_params_schema(data: Any) -> Any:
    """
    UIで扱いやすいようにparams.yamlを正規化
    """
    if isinstance(data, dict):
        if _is_param_node(data):
            normalized = {}
            for key, value in data.items():
                if key == 'choices' and isinstance(value, tuple):
                    normalized[key] = list(value)
                else:
                    normalized[key] = value
            if 'value' not in normalized:
                normalized['value'] = None
            return normalized
        else:
            return {k: normalize_params_schema(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_params_schema(item) for item in data]
    else:
        return {'value': data}


def denormalize_params_schema(data: Any) -> Any:
    """
    正規化されたparamsデータを元のスキーマに戻す
    """
    if isinstance(data, dict):
        if _is_param_node(data):
            cleaned = {}
            for key, value in data.items():
                # readonlyフラグは保存時に削除
                if key == 'readonly':
                    continue
                if key == 'choices' and isinstance(value, list):
                    cleaned[key] = [item for item in value if item not in ('', None)]
                else:
                    cleaned[key] = value
            if list(cleaned.keys()) == ['value']:
                return cleaned['value']
            return cleaned
        else:
            result = {}
            for k, v in data.items():
                # scheduler_paramsが空のオブジェクトの場合は空のまま保存
                if k == 'scheduler_params' and isinstance(v, dict) and len(v) == 0:
                    result[k] = {}
                else:
                    result[k] = denormalize_params_schema(v)
            return result
    elif isinstance(data, list):
        return [denormalize_params_schema(item) for item in data]
    else:
        return data


def normalize_augments_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    auguments.yamlをUIで扱いやすい形式に正規化
    enabledフラグを持つaugmentation/前処理を整理
    """
    if not isinstance(data, dict):
        return {}
    
    normalized = {}
    
    # image設定（そのまま）
    if 'image' in data:
        normalized['image'] = data['image']
    
    # train/val/test augmentation設定
    for split in ['train', 'val', 'test']:
        if split in data:
            split_data = data[split]
            normalized[split] = {}
            for aug_name, aug_config in split_data.items():
                if isinstance(aug_config, dict) and 'enabled' in aug_config:
                    normalized[split][aug_name] = {
                        'enabled': aug_config.get('enabled', False),
                        'params': {k: v for k, v in aug_config.items() if k != 'enabled'}
                    }
                else:
                    normalized[split][aug_name] = aug_config
    
    # preprocessing設定
    if 'preprocessing' in data:
        preproc_data = data['preprocessing']
        normalized['preprocessing'] = {}
        for proc_name, proc_config in preproc_data.items():
            if isinstance(proc_config, dict) and 'enabled' in proc_config:
                normalized['preprocessing'][proc_name] = {
                    'enabled': proc_config.get('enabled', False),
                    'params': {k: v for k, v in proc_config.items() if k != 'enabled'}
                }
            else:
                normalized['preprocessing'][proc_name] = proc_config
    
    # library設定
    if 'library' in data:
        normalized['library'] = data['library']
    
    return normalized


def denormalize_augments_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    正規化されたaugumentsデータを元のYAML形式に戻す
    """
    if not isinstance(data, dict):
        return {}
    
    denormalized = {}
    
    # image設定
    if 'image' in data:
        denormalized['image'] = data['image']
    
    # train/val/test augmentation設定
    for split in ['train', 'val', 'test']:
        if split in data:
            split_data = data[split]
            denormalized[split] = {}
            for aug_name, aug_data in split_data.items():
                if isinstance(aug_data, dict) and 'enabled' in aug_data:
                    # enabledとparamsを統合
                    aug_config = {'enabled': aug_data['enabled']}
                    if 'params' in aug_data:
                        aug_config.update(aug_data['params'])
                    denormalized[split][aug_name] = aug_config
                else:
                    denormalized[split][aug_name] = aug_data
    
    # preprocessing設定
    if 'preprocessing' in data:
        preproc_data = data['preprocessing']
        denormalized['preprocessing'] = {}
        for proc_name, proc_data in preproc_data.items():
            if isinstance(proc_data, dict) and 'enabled' in proc_data:
                proc_config = {'enabled': proc_data['enabled']}
                if 'params' in proc_data:
                    proc_config.update(proc_data['params'])
                denormalized['preprocessing'][proc_name] = proc_config
            else:
                denormalized['preprocessing'][proc_name] = proc_data
    
    # library設定
    if 'library' in data:
        denormalized['library'] = data['library']
    
    return denormalized


def remove_tunable_specs(params_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    パラメータスキーマからチューニング対象の指定を削除し、固定値のみにする
    
    Args:
        params_schema: パラメータスキーマ
    
    Returns:
        チューニング対象を削除したパラメータスキーマ
    """
    if not isinstance(params_schema, dict):
        return params_schema
    
    result = {}
    for key, value in params_schema.items():
        if key == 'optuna':
            # optunaセクションはそのまま保持
            result[key] = value
        elif isinstance(value, dict):
            # typeプロパティがある場合はチューニング対象として扱う
            if 'type' in value:
                # valueがある場合はvalueのみを保持、ない場合はdict全体を保持
                if 'value' in value:
                    result[key] = value.get('value')
                else:
                    # valueがない場合は、type以外のプロパティを保持
                    cleaned = {k: v for k, v in value.items() if k != 'type' and k not in PARAM_META_KEYS}
                    if len(cleaned) == 1 and 'value' in cleaned:
                        result[key] = cleaned['value']
                    else:
                        result[key] = remove_tunable_specs(cleaned)
            else:
                # typeがない場合は再帰的に処理
                result[key] = remove_tunable_specs(value)
        elif isinstance(value, list):
            result[key] = [remove_tunable_specs(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    return result


def get_available_models() -> List[str]:
    """
    利用可能なモデルのリストを取得（model_factory.pyのMODEL_REGISTRYから）
    
    Returns:
        モデル名のリスト
    """
    try:
        project_root = get_project_root()
        # sys.pathにプロジェクトルートを追加
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.models.model_factory import list_available_models
        return list_available_models()
    except Exception as e:
        logger.error(f"利用可能なモデルリストの取得に失敗しました: {e}")
        # フォールバック: デフォルトのモデルリストを返す
        return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

