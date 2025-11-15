@echo off
REM SliceBench Windows Setup and Run Script

echo ============================================
echo SliceBench Setup (Windows)
echo ============================================

REM Create venv if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet opencv-python scikit-learn matplotlib seaborn pandas tqdm

REM Download data if needed
if not exist "data\slicebench\metadata.json" (
    echo Downloading sample images and generating dataset...
    python scripts\download_real_samples.py
)

REM Run evaluations
echo.
echo Running ResNet50 evaluation...
python scripts\run_evaluation.py --model resnet50

echo.
echo Running ViT-B/16 evaluation...
python scripts\run_evaluation.py --model vit_b_16

REM Generate visualizations
echo.
echo Generating visualizations...
python scripts\visualize.py

REM Show results
echo.
echo ============================================
echo Results Summary
echo ============================================
type results\summary_report.txt

echo.
echo ============================================
echo Complete! Check results\ folder for details.
echo ============================================
pause
