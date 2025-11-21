#!/bin/bash
#
# MLflow Tracking Serverを起動するスクリプト
#
# 使用方法:
#   ./scripts/start_mlflow.sh              # フォアグラウンドで起動
#   ./scripts/start_mlflow.sh --background # バックグラウンドで起動
#   ./scripts/start_mlflow.sh --stop       # 停止
#

set -e

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# ディレクトリの作成
mkdir -p experiments

# config.yamlから設定を読み込む（yamlがない場合はgrepで簡易パース）
read_config() {
    # まずPython + yamlで試行
    if command -v python3 &> /dev/null; then
        RESULT=$(python3 -c "
try:
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    mlflow_config = config.get('mlflow', {})
    server_config = mlflow_config.get('server', {})
    print(f\"{server_config.get('host', '0.0.0.0')}:{server_config.get('port', 5001)}\")
except:
    print('FALLBACK')
" 2>/dev/null)
        
        if [ "$RESULT" != "FALLBACK" ] && [ -n "$RESULT" ]; then
            echo "$RESULT"
            return
        fi
    fi
    
    # フォールバック: grepで簡易パース
    if [ -f "config.yaml" ]; then
        HOST=$(grep -A 2 "server:" config.yaml | grep "host:" | awk '{print $2}' || echo "0.0.0.0")
        PORT=$(grep -A 2 "server:" config.yaml | grep "port:" | awk '{print $2}' || echo "5001")
        echo "${HOST}:${PORT}"
    else
        echo "0.0.0.0:5001"
    fi
}

# constants.py（MLFLOW_UI_URL）から設定を取得
read_constants_url() {
    if command -v python3 &> /dev/null; then
        RESULT=$(python3 <<'PY'
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src" / "web"))

try:
    from data_management.constants import MLFLOW_UI_URL
    if MLFLOW_UI_URL:
        print(MLFLOW_UI_URL)
except Exception:
    pass
PY
)
        if [ -n "$RESULT" ]; then
            echo "$RESULT"
            return
        fi
    fi
    echo ""
}

parse_host_port_from_url() {
    URL="$1"
    if [ -z "$URL" ]; then
        echo ""
        return
    fi
    python3 <<PY
from urllib.parse import urlparse
import sys

url = "$URL"
if "://" not in url:
    url = "http://" + url
parsed = urlparse(url)
host = parsed.hostname or "0.0.0.0"
port = parsed.port or 5001
print(f"{host}:{port}")
PY
}

CONSTANTS_URL=$(read_constants_url)

if [ -n "$CONSTANTS_URL" ]; then
    HOST_PORT=$(parse_host_port_from_url "$CONSTANTS_URL")
else
    HOST_PORT=$(read_config)
fi

CONFIG_HOST=$(echo "$HOST_PORT" | cut -d: -f1)
CONFIG_PORT=$(echo "$HOST_PORT" | cut -d: -f2)

# デフォルト設定（config.yamlから読み込んだ値を使用）
BACKEND_STORE="sqlite:///experiments/mlflow.db"
ARTIFACT_ROOT="experiments/mlruns"
HOST="${CONFIG_HOST}"
PORT="${CONFIG_PORT}"

# ヘルプメッセージ
show_help() {
    echo "MLflow Tracking Server 起動スクリプト"
    echo ""
    echo "使用方法:"
    echo "  ./scripts/start_mlflow.sh [オプション]"
    echo ""
    echo "オプション:"
    echo "  --background, -b     バックグラウンドで起動"
    echo "  --stop, -s           MLflow Serverを停止"
    echo "  --port PORT          ポート番号を指定（config.yamlを上書き）"
    echo "  --help, -h           このヘルプを表示"
    echo ""
    echo "デフォルト設定:"
    echo "  config.yamlの mlflow.server.host と mlflow.server.port を使用"
    echo "  現在の設定: host=${CONFIG_HOST}, port=${CONFIG_PORT}"
    echo ""
    echo "例:"
    echo "  ./scripts/start_mlflow.sh              # config.yamlの設定で起動"
    echo "  ./scripts/start_mlflow.sh -b           # バックグラウンドで起動"
    echo "  ./scripts/start_mlflow.sh --port 5002  # ポート5002で起動"
    echo "  ./scripts/start_mlflow.sh --stop       # 停止"
}

# MLflow Serverを停止
stop_mlflow() {
    echo "MLflow Serverを停止します..."
    
    # プロセスを検索
    PID=$(ps aux | grep "mlflow server" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PID" ]; then
        echo "MLflow Serverは起動していません"
        exit 0
    fi
    
    # プロセスを終了
    kill $PID
    echo "MLflow Server (PID: $PID) を停止しました"
    exit 0
}

# 引数の解析
BACKGROUND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -s|--stop)
            stop_mlflow
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# MLflow Serverが既に起動しているか確認
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "エラー: ポート $PORT は既に使用されています"
    echo ""
    echo "以下のコマンドで確認してください:"
    echo "  lsof -i :$PORT"
    echo ""
    echo "MLflow Serverを停止する場合:"
    echo "  ./scripts/start_mlflow.sh --stop"
    exit 1
fi

# MLflow Serverを起動
echo "========================================"
echo "MLflow Tracking Server を起動します"
echo "========================================"
echo "設定: config.yamlから読み込み"
echo "Backend Store: $BACKEND_STORE"
echo "Artifact Root: $ARTIFACT_ROOT"
echo "Host: $HOST"
echo "Port: $PORT"
echo "========================================"

if [ "$BACKGROUND" = true ]; then
    # バックグラウンドで起動
    echo "バックグラウンドで起動中..."
    nohup mlflow server \
        --backend-store-uri "$BACKEND_STORE" \
        --default-artifact-root "$ARTIFACT_ROOT" \
        --host "$HOST" \
        --port "$PORT" > mlflow.log 2>&1 &
    
    PID=$!
    echo ""
    echo "✓ MLflow Server を起動しました（PID: $PID）"
    echo ""
    echo "アクセス: http://localhost:$PORT"
    echo "ログファイル: mlflow.log"
    echo ""
    echo "停止する場合:"
    echo "  ./scripts/start_mlflow.sh --stop"
    echo "  または"
    echo "  kill $PID"
else
    # フォアグラウンドで起動
    echo ""
    echo "アクセス: http://localhost:$PORT"
    echo ""
    echo "停止する場合: Ctrl+C"
    echo "========================================"
    echo ""
    
    mlflow server \
        --backend-store-uri "$BACKEND_STORE" \
        --default-artifact-root "$ARTIFACT_ROOT" \
        --host "$HOST" \
        --port "$PORT"
fi

