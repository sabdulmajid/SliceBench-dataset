"""Verify that model predictions actually work on generated images."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from src.models import ModelWrapper
from src.data_utils import ImageProcessor

model = ModelWrapper('resnet50')
proc = ImageProcessor()

sample_dir = Path('data/sample_source/samples')
samples = sorted(list(sample_dir.glob('*.jpg')))[:3]

print("Testing model predictions on generated images:\n")
print("="*70)

for sample_path in samples:
    img = Image.open(sample_path)
    tensor = proc.preprocess(img)
    
    probs, indices = model.predict(tensor.unsqueeze(0), top_k=5)
    
    expected_label = None
    if 'tabby_cat' in sample_path.name:
        expected_label = 281
    elif 'golden_retriever' in sample_path.name:
        expected_label = 207
    elif 'giant_panda' in sample_path.name:
        expected_label = 388
    
    print(f"\nImage: {sample_path.name}")
    print(f"Expected class: {expected_label}")
    print(f"\nTop 5 predictions:")
    
    correct = False
    for i in range(5):
        pred_class = indices[0, i].item()
        pred_prob = probs[0, i].item()
        marker = " ✓" if pred_class == expected_label else ""
        print(f"  {i+1}. Class {pred_class:3d}: {pred_prob:6.1%}{marker}")
        if pred_class == expected_label:
            correct = True
    
    if correct:
        print(f"  → CORRECT in top-5!")
    else:
        print(f"  → INCORRECT (expected {expected_label} not in top-5)")
    
    print("-"*70)

print("\n" + "="*70)
print("Note: Synthetic images may not match ImageNet perfectly.")
print("This is expected - we're testing for bias detection, not accuracy.")
