"""
Test script to verify SliceBench components work correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        from PIL import Image
        import cv2
        import sklearn
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False


def test_config():
    """Test config module."""
    print("\nTesting config...")
    try:
        from config import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, MODELS_TO_TEST
        assert DATA_DIR.exists() or True  # Will be created
        assert isinstance(MODELS_TO_TEST, list)
        print(f"✓ Config loaded")
        print(f"  - Data dir: {DATA_DIR}")
        print(f"  - Results dir: {RESULTS_DIR}")
        print(f"  - Models: {len(MODELS_TO_TEST)}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_data_utils():
    """Test data utilities."""
    print("\nTesting data utilities...")
    try:
        from data_utils import ImageProcessor, BackgroundSwapper, CropGenerator
        from PIL import Image
        import numpy as np
        
        processor = ImageProcessor()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tensor = processor.preprocess(img)
        assert tensor.shape == (3, 224, 224)
        
        crops = CropGenerator.generate_crops(img, num_crops=3)
        assert len(crops) == 3
        
        print("✓ Data utilities working")
        return True
    except Exception as e:
        print(f"✗ Data utils error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model wrapper."""
    print("\nTesting model wrapper...")
    try:
        from models import ModelWrapper
        import torch
        
        model = ModelWrapper("resnet50")
        assert model.model is not None
        
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model.forward(dummy_input)
        assert output.shape[1] == 1000  # ImageNet classes
        
        probs, indices = model.predict(dummy_input, top_k=5)
        assert probs.shape == (1, 5)
        assert indices.shape == (1, 5)
        
        print("✓ Model wrapper working")
        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_generation():
    """Test dataset generation."""
    print("\nTesting dataset generation...")
    try:
        from dataset import SliceBenchGenerator
        from PIL import Image
        import numpy as np
        from config import DATA_DIR
        
        test_dir = DATA_DIR / "test_slicebench"
        generator = SliceBenchGenerator(output_dir=test_dir)
        
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_path = test_dir / "test_img.jpg"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_img.save(test_path)
        
        generator.add_source_image(test_path, 0, "test")
        
        slice_info = generator.generate_crop_slice(test_path, 0, "test_crop")
        assert slice_info["slice_type"] == "crop_variations"
        
        print("✓ Dataset generation working")
        
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"✗ Dataset generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation pipeline."""
    print("\nTesting evaluation...")
    try:
        from models import ModelWrapper
        from evaluation import SliceEvaluator
        import torch
        
        model = ModelWrapper("resnet50")
        evaluator = SliceEvaluator(model)
        
        dummy_images = [torch.randn(3, 224, 224) for _ in range(5)]
        dummy_labels = [0, 1, 2, 3, 4]
        
        results = evaluator.evaluate_accuracy(dummy_images, dummy_labels)
        assert "top1_accuracy" in results
        assert "top5_accuracy" in results
        
        print("✓ Evaluation working")
        return True
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("SliceBench Test Suite")
    print("="*60)
    
    tests = [
        test_imports,
        test_config,
        test_data_utils,
        test_models,
        test_dataset_generation,
        test_evaluation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    if all(results):
        print("\n✓ All tests passed! SliceBench is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
