"""
アプリ内で利用する定数
"""
"""
アプリ内で利用する定数
"""
import os

try:
    from django.conf import settings  # Django環境下のみ
except Exception:  # pragma: no cover - Django未設定時
    settings = None

# MLflow UI のデフォルトURL（ポート5001を使用）
MLFLOW_UI_DEFAULT_URL = os.environ.get("MLFLOW_UI_URL", "http://127.0.0.1:5001")


def _get_mlflow_ui_url() -> str:
    """
    Django settings または環境変数からMLflow UIのURLを取得
    """
    if settings is not None:
        return getattr(settings, "MLFLOW_UI_URL", MLFLOW_UI_DEFAULT_URL)
    return MLFLOW_UI_DEFAULT_URL


MLFLOW_UI_URL = _get_mlflow_ui_url()
print(MLFLOW_UI_URL)

