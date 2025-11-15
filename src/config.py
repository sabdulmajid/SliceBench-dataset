from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_IMAGE_SIZE = 224

MODELS_TO_TEST = [
    "resnet50",
    "resnet101",
    "vit_b_16",
    "efficientnet_b0",
    "mobilenet_v3_large",
]

SLICE_TYPES = [
    "background_swap",
    "crop_variations",
    "context_removal",
    "watermark_test",
    "texture_bias",
]
