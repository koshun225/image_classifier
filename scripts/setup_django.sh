#!/bin/bash
# Django環境のセットアップスクリプト

set -e

echo "========================================="
echo "Django環境のセットアップを開始します"
echo "========================================="

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# 1. 依存パッケージのインストール
echo ""
echo "[1/5] 依存パッケージをインストールしています..."
pip install -r requirements.txt

# 2. Djangoプロジェクトのディレクトリに移動
cd src/web

# 3. マイグレーションファイルの作成
echo ""
echo "[2/5] マイグレーションファイルを作成しています..."
python manage.py makemigrations

# 4. マイグレーションを適用
echo ""
echo "[3/5] マイグレーションを適用しています..."
python manage.py migrate

# 5. 静的ファイルを収集
echo ""
echo "[4/5] 静的ファイルを収集しています..."
python manage.py collectstatic --noinput --clear

# 6. スーパーユーザーの作成（オプション）
echo ""
echo "[5/5] スーパーユーザーを作成しますか？ (y/n)"
read -p "> " answer

if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    python manage.py createsuperuser
fi

echo ""
echo "========================================="
echo "Django環境のセットアップが完了しました！"
echo "========================================="
echo ""
echo "管理画面にアクセスするには、以下のコマンドを実行してください："
echo "  cd src/web"
echo "  python manage.py runserver"
echo ""
echo "その後、ブラウザで http://127.0.0.1:8000/admin/ にアクセスしてください。"
echo ""

