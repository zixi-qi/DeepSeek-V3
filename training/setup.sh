#!/bin/bash

# Setup script for DeepSeek V3 training environment

echo "==========================================="
echo "DeepSeek V3 Training Environment Setup"
echo "==========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv deepseek_env
source deepseek_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Detected CUDA version: $cuda_version"
    
    # Install PyTorch based on CUDA version
    if [[ "$cuda_version" == "12.1" ]] || [[ "$cuda_version" == "12.2" ]] || [[ "$cuda_version" == "12.3" ]]; then
        echo "Installing PyTorch for CUDA 12.x..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$cuda_version" == "11.8" ]] || [[ "$cuda_version" == "11.7" ]]; then
        echo "Installing PyTorch for CUDA 11.x..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    else
        echo "WARNING: Unsupported CUDA version. Installing CPU-only PyTorch..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    fi
else
    echo "No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Fix numpy compatibility
echo "Fixing numpy compatibility..."
pip install "numpy<2.0"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
if command -v nvidia-smi &> /dev/null; then
    python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
fi
python -c "import transformers; print(f'✓ Transformers {transformers.__version__} installed')"
python -c "import accelerate; print(f'✓ Accelerate {accelerate.__version__} installed')"
python -c "import datasets; print(f'✓ Datasets {datasets.__version__} installed')"

echo ""
echo "Running setup test..."
python test_setup.py

echo ""
echo "==========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source deepseek_env/bin/activate"
echo ""
echo "To start training:"
echo "  # For tiny model (fits on most GPUs):"
echo "  ./train_tiny.sh"
echo ""
echo "  # For full mini model (requires 80GB GPU):"
echo "  python train.py"
echo "==========================================="
