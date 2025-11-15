"""
Generate case study comparing ResNet50 and ViT-B/16 on SliceBench.
Demonstrates bias differences between CNN and Transformer architectures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd

from src.config import RESULTS_DIR


def generate_case_study():
    """Generate comparative case study between ResNet and ViT."""
    
    resnet_file = RESULTS_DIR / "resnet50_results.json"
    vit_file = RESULTS_DIR / "vit_b_16_results.json"
    
    if not resnet_file.exists() or not vit_file.exists():
        print("Missing results. Run: python run_evaluation.py --model resnet50")
        print("                      python run_evaluation.py --model vit_b_16")
        return
    
    with open(resnet_file) as f:
        resnet = json.load(f)
    with open(vit_file) as f:
        vit = json.load(f)
    
    output_file = RESULTS_DIR / "case_study_resnet_vs_vit.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CASE STUDY: ResNet50 vs ViT-B/16 Bias Analysis\n")
        f.write("="*80 + "\n\n")
        
        f.write("BASELINE PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Confidence':<12}\n")
        f.write(f"{'ResNet50':<20} {resnet['baseline']['top1_accuracy']:<12.1%} "
                f"{resnet['baseline']['top5_accuracy']:<12.1%} "
                f"{resnet['baseline']['mean_confidence']:<12.1%}\n")
        f.write(f"{'ViT-B/16':<20} {vit['baseline']['top1_accuracy']:<12.1%} "
                f"{vit['baseline']['top5_accuracy']:<12.1%} "
                f"{vit['baseline']['mean_confidence']:<12.1%}\n\n")
        
        f.write("ROBUSTNESS BY SLICE TYPE\n")
        f.write("-"*80 + "\n")
        
        slice_types = ['bg', 'crop', 'ctx', 'wm', 'tex']
        slice_names = {
            'bg': 'Background Swap',
            'crop': 'Crop Variations',
            'ctx': 'Context Removal',
            'wm': 'Watermark',
            'tex': 'Texture/Color Bias'
        }
        
        for slice_type in slice_types:
            resnet_accs = [s['top1_accuracy'] for s in resnet['slice_results'] 
                          if s['slice_name'].startswith(slice_type)]
            vit_accs = [s['top1_accuracy'] for s in vit['slice_results'] 
                       if s['slice_name'].startswith(slice_type)]
            
            if resnet_accs and vit_accs:
                resnet_avg = sum(resnet_accs) / len(resnet_accs)
                vit_avg = sum(vit_accs) / len(vit_accs)
                
                resnet_drop = resnet['baseline']['top1_accuracy'] - resnet_avg
                vit_drop = vit['baseline']['top1_accuracy'] - vit_avg
                
                f.write(f"\n{slice_names[slice_type]}\n")
                f.write(f"  ResNet50: {resnet_avg:5.1%} (drop: {resnet_drop:+5.1%})\n")
                f.write(f"  ViT-B/16: {vit_avg:5.1%} (drop: {vit_drop:+5.1%})\n")
                
                if abs(resnet_drop - vit_drop) > 0.05:
                    more_robust = "ViT-B/16" if abs(vit_drop) < abs(resnet_drop) else "ResNet50"
                    f.write(f"  → {more_robust} is more robust\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        resnet_biases = [b for b in resnet.get('bias_detections', []) 
                        if b.get('has_significant_bias', False)]
        vit_biases = [b for b in vit.get('bias_detections', []) 
                     if b.get('has_significant_bias', False)]
        
        f.write(f"1. ResNet50 shows {len(resnet_biases)} significant biases\n")
        f.write(f"2. ViT-B/16 shows {len(vit_biases)} significant biases\n\n")
        
        if resnet['baseline']['top1_accuracy'] > vit['baseline']['top1_accuracy']:
            f.write("3. ResNet50 achieves higher baseline accuracy\n")
        else:
            f.write("3. ViT-B/16 achieves higher baseline accuracy\n")
        
        f.write("\n4. Architecture differences:\n")
        f.write("   - CNNs (ResNet) have stronger inductive biases\n")
        f.write("   - Transformers (ViT) learn from data more flexibly\n")
        f.write("   - Both show some reliance on context/background\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nCase study saved to: {output_file}")
    
    with open(output_file) as f:
        print(f.read())


if __name__ == "__main__":
    generate_case_study()
