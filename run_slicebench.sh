#!/bin/bash
# SliceBench Linux/Mac Setup and Run Script

set -e

echo "============================================"
echo "SliceBench Setup (Linux/Mac with NVIDIA GPU)"
echo "============================================"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
python -m pip install --quiet --upgrade pip

# Install PyTorch with CUDA support
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install --quiet opencv-python scikit-learn matplotlib seaborn pandas tqdm

# Download data if needed
if [ ! -f "data/slicebench/metadata.json" ]; then
    echo "Downloading sample images and generating dataset..."
    python download_real_samples.py
fi

# Run evaluations
echo ""
echo "Running ResNet50 evaluation..."
python run_evaluation.py --model resnet50

echo ""
echo "Running ViT-B/16 evaluation..."
python run_evaluation.py --model vit_b_16

# Generate visualizations
echo ""
echo "Generating visualizations..."
python visualize.py

# Show results
echo ""
echo "============================================"
echo "Results Summary"
echo "============================================"
cat results/summary_report.txt

echo ""
echo "============================================"
echo "Complete! Check results/ folder for details."
echo "============================================"
