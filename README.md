# SliceBench: Systematic Bias Probing for ImageNet Models

A lightweight, research-oriented framework for testing robustness and biases of pre-trained ImageNet models through controlled test slices.

## Overview

SliceBench systematically probes vision models for shortcuts and biases including:
- **Background leakage**: Does the model rely on backgrounds vs objects?
- **Position bias**: Does performance degrade when objects aren't centered?
- **Context reliance**: Can the model recognize objects without context?
- **Watermark sensitivity**: Do spurious correlations affect predictions?
- **Texture bias**: Does color/texture dominate over shape?

## Key Features

- **Automated slice generation**: Create controlled test scenarios from source images
- **Multi-model evaluation**: Test multiple architectures in parallel
- **Attribution analysis**: GradCAM-based attention pattern analysis
- **Calibration metrics**: Beyond accuracy - measure confidence calibration
- **Comprehensive reporting**: JSON results + summary visualizations

## Installation

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Direct Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- OpenCV
- scikit-learn
- matplotlib, seaborn

## Quick Start

### One-Command Setup

**Windows:**
```bash
run_slicebench.bat
```

**Linux/Mac (with NVIDIA GPU):**
```bash
chmod +x run_slicebench.sh
./run_slicebench.sh
```

These scripts automatically:
1. Create virtual environment
2. Install dependencies
3. Download sample images
4. Run evaluations (ResNet50 + ViT-B/16)
5. Generate visualizations and case study

### Manual Installation (Optional)

If you prefer step-by-step control:

If you prefer step-by-step control:

**1. Setup Environment:**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Generate Dataset:**

```bash
python download_real_samples.py
```

**3. Run Evaluation:**

```bash
# Single model
python run_evaluation.py --model resnet50

# Multiple models
python run_evaluation.py --model vit_b_16
```

**4. Generate Visualizations:**

```bash
python visualize.py
```

## Case Study: ResNet50 vs ViT-B/16

The included case study compares CNN (ResNet) and Transformer (ViT) architectures:

```bash
python generate_case_study.py
```

**Key Findings:**
- **Baseline:** ResNet50 achieves 80% vs ViT-B/16 60% top-1 accuracy
- **Background Swap:** ResNet50 more robust (0% drop vs 7% drop)
- **Context Removal:** ViT-B/16 more robust (0% drop vs 10% drop)
- **Confidence:** ViT shows higher confidence (72% vs 40%) despite lower accuracy
- **Bias Detection:** ViT exhibits more significant biases (2 vs 1)

Results demonstrate architectural differences:
- CNNs have stronger inductive biases for spatial structure
- Transformers are more flexible but require more data
- Both architectures show context/background reliance

See `results/case_study_resnet_vs_vit.txt` for full analysis.

## Usage with Your Own Images

```python
from pathlib import Path
from dataset import SliceBenchGenerator

generator = SliceBenchGenerator()

# Add your images
image_path = Path("path/to/your/image.jpg")
label = 123  # ImageNet class ID
class_name = "cat"

# Generate test slices
generator.generate_background_swap_slice(
    image_path, label, background_paths, "cat_bg_test"
)
generator.generate_crop_slice(image_path, label, "cat_crop_test")
generator.generate_context_removal_slice(image_path, label, "cat_ctx_test")

generator.save_metadata()
```

### Custom Evaluation

```python
from models import ModelWrapper
from evaluation import SliceEvaluator, BiasDetector
from dataset import SliceBenchLoader

# Load model
model = ModelWrapper("resnet50")
evaluator = SliceEvaluator(model)

# Load data
loader = SliceBenchLoader()
images, labels, info = loader.load_slice("cat_bg_test")

# Evaluate
results = evaluator.evaluate_slice(images, labels, "cat_bg_test")
print(f"Accuracy: {results['top1_accuracy']:.3f}")

# Detect bias
baseline_imgs, baseline_labels = loader.get_baseline_images()
bias = BiasDetector.detect_background_bias(
    evaluator, baseline_imgs, images, labels
)
print(f"Background bias detected: {bias['has_significant_bias']}")
```

## Test Slice Types

### 1. Background Swap
Replace image backgrounds to test if model relies on background vs object.

### 2. Crop Variations
Test different crop scales and positions to detect object localization bias.

### 3. Context Removal
Blur or isolate objects to test context vs object reliance.

### 4. Watermark Test
Add watermarks in different positions to test spurious correlations.

### 5. Texture Bias
Add colored patches to test if model is biased toward texture/color.

## Evaluation Metrics

- **Accuracy**: Top-1 and Top-5 classification accuracy
- **Calibration**: Brier score and Expected Calibration Error (ECE)
- **Confidence**: Mean prediction confidence and variance
- **Attribution**: GradCAM-based attention pattern analysis

## Output

### JSON Results

Detailed per-model results saved to `results/<model_name>_results.json`:

```json
{
  "model_name": "resnet50",
  "baseline": {
    "top1_accuracy": 0.85,
    "top5_accuracy": 0.95,
    "mean_confidence": 0.78
  },
  "slice_results": [...],
  "bias_detections": [
    {
      "bias_type": "background_reliance",
      "accuracy_drop": 0.15,
      "has_significant_bias": true,
      "severity": "high"
    }
  ]
}
```

### Summary Report

Human-readable summary in `results/summary_report.txt`:

```
Model           Baseline  Num BG Biases  Avg BG Drop  Num Ctx Biases  Avg Ctx Drop
resnet50        0.850     3              0.123        2               0.089
efficientnet_b0 0.832     2              0.098        1               0.067
```

## Supported Models

Default models (from torchvision):
- ResNet-50, ResNet-101
- EfficientNet-B0
- MobileNet-V3-Large
- ViT-B/16

Add custom models by extending `ModelWrapper` in `models.py`.

## Configuration

Edit `config.py` to customize:
- Image size and normalization
- Models to test
- Slice types to generate
- Output directories

## Performance Notes

- Uses GPU automatically if available
- GradCAM computation is expensive - runs on subset for speed
- Larger datasets may require batching (not implemented in basic version)

## Limitations

- Background segmentation uses GrabCut - may fail on complex images
- Designed for ImageNet-style images (224x224)
- Attribution tested primarily on CNN architectures
- Sample dataset is small - use your own images for comprehensive evaluation
