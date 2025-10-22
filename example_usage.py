"""
Example usage of SliceBench for testing a single model.
"""

import torch
from pathlib import Path
from models import ModelWrapper
from evaluation import SliceEvaluator
from dataset import SliceBenchLoader
from config import DATA_DIR


def quick_test_example():
    """Quick test of a single model on SliceBench."""
    
    print("Loading model...")
    model = ModelWrapper("resnet50")
    
    print("Loading SliceBench data...")
    loader = SliceBenchLoader(DATA_DIR / "slicebench")
    
    print("Creating evaluator...")
    evaluator = SliceEvaluator(model)
    
    print("\nGetting baseline images...")
    baseline_images, baseline_labels = loader.get_baseline_images()
    print(f"Loaded {len(baseline_images)} baseline images")
    
    print("\nEvaluating baseline performance...")
    baseline_results = evaluator.evaluate_slice(
        baseline_images, baseline_labels, "baseline"
    )
    
    print(f"Baseline Top-1 Accuracy: {baseline_results['top1_accuracy']:.3f}")
    print(f"Baseline Top-5 Accuracy: {baseline_results['top5_accuracy']:.3f}")
    print(f"Mean Confidence: {baseline_results['mean_confidence']:.3f}")
    
    print("\nTesting background swap slices...")
    bg_slices = loader.load_all_slices(slice_type="background_swap")
    
    if bg_slices:
        images, labels, info = bg_slices[0]
        results = evaluator.evaluate_slice(images, labels, info["slice_id"])
        
        print(f"Background Swap Accuracy: {results['top1_accuracy']:.3f}")
        print(f"Accuracy Drop: {baseline_results['top1_accuracy'] - results['top1_accuracy']:.3f}")
    
    print("\nTesting context removal slices...")
    ctx_slices = loader.load_all_slices(slice_type="context_removal")
    
    if ctx_slices:
        images, labels, info = ctx_slices[0]
        results = evaluator.evaluate_slice(images, labels, info["slice_id"])
        
        print(f"Context Removal Accuracy: {results['top1_accuracy']:.3f}")
        print(f"Accuracy Drop: {baseline_results['top1_accuracy'] - results['top1_accuracy']:.3f}")
    
    print("\n✓ Quick test completed!")


if __name__ == "__main__":
    quick_test_example()
