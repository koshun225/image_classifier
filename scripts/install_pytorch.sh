#!/bin/bash
# PyTorchインストールスクリプト

echo "================================================"
echo "PyTorchをインストールします"
echo "================================================"

# Anaconda環境の場合
if command -v conda &> /dev/null; then
    echo ""
    echo "✓ Conda環境を検出しました"
    echo ""
    echo "PyTorchのインストール方法を選択してください:"
    echo "1. CPU版（軽量、すぐに使える）"
    echo "2. GPU版（CUDA対応、高速）"
    echo ""
    read -p "選択 (1 or 2): " choice
    
    if [ "$choice" == "1" ]; then
        echo ""
        echo "CPU版PyTorchをインストール中..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    elif [ "$choice" == "2" ]; then
        echo ""
        echo "GPU版PyTorchをインストール中..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "無効な選択です"
        exit 1
    fi
else
    # pip環境の場合
    echo ""
    echo "✓ pip環境を検出しました"
    echo ""
    echo "CPU版PyTorchをインストール中..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "================================================"
echo "インストール確認"
echo "================================================"
python -c "import torch; import torchvision; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ TorchVision version: {torchvision.__version__}')"

echo ""
echo "================================================"
echo "PyTorch Lightningもインストールしますか？ (y/n)"
echo "================================================"
read -p "選択: " install_lightning

if [ "$install_lightning" == "y" ]; then
    echo ""
    echo "PyTorch Lightningをインストール中..."
    pip install pytorch-lightning
    python -c "import pytorch_lightning as pl; print(f'✓ PyTorch Lightning version: {pl.__version__}')"
fi

echo ""
echo "================================================"
echo "✓ インストール完了！"
echo "================================================"
echo ""
echo "次のステップ:"
echo "1. データセットをテスト:"
echo "   pytest tests/test_pytorch_dataset.py -v -s"
echo ""
echo "2. 学習を開始:"
echo "   python scripts/train.py --theme-id 7"
echo ""

