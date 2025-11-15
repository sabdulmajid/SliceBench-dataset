import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.config import DATA_DIR, MODELS_TO_TEST, RESULTS_DIR
from src.dataset import SliceBenchLoader
from src.models import ModelWrapper
from src.evaluation import SliceEvaluator, BiasDetector, save_results


def evaluate_model_on_slicebench(model_name: str, 
                                 data_dir: Path = DATA_DIR / "slicebench",
                                 output_dir: Path = RESULTS_DIR):
    """Run full SliceBench evaluation on a model."""
    print(f"\nEvaluating {model_name}...")
    
    model = ModelWrapper(model_name)
    evaluator = SliceEvaluator(model)
    loader = SliceBenchLoader(data_dir)
    
    all_results = {
        "model_name": model_name,
        "slice_results": [],
        "bias_detections": [],
    }
    
    print("Loading baseline images...")
    baseline_images, baseline_labels = loader.get_baseline_images()
    
    baseline_metrics = evaluator.evaluate_slice(
        baseline_images, baseline_labels, "baseline"
    )
    all_results["baseline"] = baseline_metrics
    
    print("\nEvaluating background swap slices...")
    bg_slices = loader.load_all_slices(slice_type="background_swap")
    for images, labels, info in tqdm(bg_slices, desc="Background"):
        result = evaluator.evaluate_slice(images, labels, info["slice_id"])
        all_results["slice_results"].append(result)
        
        orig_images = [baseline_images[i] for i, s in enumerate(loader.metadata["slices"]) 
                       if s["slice_id"] == info["slice_id"]]
        orig_labels = [baseline_labels[i] for i, s in enumerate(loader.metadata["slices"]) 
                      if s["slice_id"] == info["slice_id"]]
        
        if orig_images:
            bias = BiasDetector.detect_background_bias(
                evaluator, orig_images, images, labels
            )
            all_results["bias_detections"].append(bias)
    
    print("\nEvaluating crop variation slices...")
    crop_slices = loader.load_all_slices(slice_type="crop_variations")
    for images, labels, info in tqdm(crop_slices, desc="Crops"):
        result = evaluator.evaluate_slice(images, labels, info["slice_id"])
        all_results["slice_results"].append(result)
    
    print("\nEvaluating context removal slices...")
    context_slices = loader.load_all_slices(slice_type="context_removal")
    for images, labels, info in tqdm(context_slices, desc="Context"):
        result = evaluator.evaluate_slice(images, labels, info["slice_id"])
        all_results["slice_results"].append(result)
        
        orig_images = [baseline_images[i] for i, s in enumerate(loader.metadata["slices"]) 
                       if s["slice_id"] == info["slice_id"]]
        orig_labels = [baseline_labels[i] for i, s in enumerate(loader.metadata["slices"]) 
                      if s["slice_id"] == info["slice_id"]]
        
        if orig_images:
            bias = BiasDetector.detect_context_reliance(
                evaluator, orig_images, images, labels
            )
            all_results["bias_detections"].append(bias)
    
    print("\nEvaluating watermark slices...")
    watermark_slices = loader.load_all_slices(slice_type="watermark_test")
    for images, labels, info in tqdm(watermark_slices, desc="Watermarks"):
        result = evaluator.evaluate_slice(images, labels, info["slice_id"])
        all_results["slice_results"].append(result)
    
    print("\nEvaluating texture bias slices...")
    texture_slices = loader.load_all_slices(slice_type="texture_bias")
    for images, labels, info in tqdm(texture_slices, desc="Texture"):
        result = evaluator.evaluate_slice(images, labels, info["slice_id"])
        all_results["slice_results"].append(result)
    
    output_path = output_dir / f"{model_name}_results.json"
    save_results(all_results, output_path)
    print(f"\nResults saved to {output_path}")
    
    return all_results


def generate_summary_report(results_dir: Path = RESULTS_DIR, 
                           output_file: Path = None):
    """Generate summary report across all evaluated models."""
    import json
    import pandas as pd
    
    if output_file is None:
        output_file = results_dir / "summary_report.txt"
    
    result_files = list(results_dir.glob("*_results.json"))
    
    if not result_files:
        print("No result files found")
        return
    
    summary_data = []
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        model_name = results["model_name"]
        baseline = results["baseline"]
        
        bg_biases = [b for b in results["bias_detections"] if b["bias_type"] == "background_reliance"]
        ctx_biases = [b for b in results["bias_detections"] if b["bias_type"] == "context_reliance"]
        
        summary_data.append({
            "model": model_name,
            "baseline_top1": baseline["top1_accuracy"],
            "baseline_top5": baseline["top5_accuracy"],
            "num_bg_biases": sum(1 for b in bg_biases if b["has_significant_bias"]),
            "avg_bg_drop": sum(b["accuracy_drop"] for b in bg_biases) / len(bg_biases) if bg_biases else 0,
            "num_ctx_biases": sum(1 for b in ctx_biases if b["has_significant_bias"]),
            "avg_ctx_drop": sum(b["accuracy_drop"] for b in ctx_biases) / len(ctx_biases) if ctx_biases else 0,
        })
    
    df = pd.DataFrame(summary_data)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SliceBench Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 80 + "\n")
        
        best_model = df.loc[df["baseline_top1"].idxmax(), "model"]
        f.write(f"Best baseline accuracy: {best_model}\n")
        
        most_robust_bg = df.loc[df["avg_bg_drop"].idxmin(), "model"]
        f.write(f"Most robust to background changes: {most_robust_bg}\n")
        
        most_robust_ctx = df.loc[df["avg_ctx_drop"].idxmin(), "model"]
        f.write(f"Most robust to context removal: {most_robust_ctx}\n")
    
    print(f"\nSummary report saved to {output_file}")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run SliceBench evaluation")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to evaluate (default: all)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR / "slicebench"),
                       help="Path to SliceBench data")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                       help="Output directory for results")
    parser.add_argument("--summary", action="store_true",
                       help="Generate summary report only")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.summary:
        generate_summary_report(output_dir)
        return
    
    models = [args.model] if args.model else MODELS_TO_TEST
    
    for model_name in models:
        try:
            evaluate_model_on_slicebench(
                model_name, 
                Path(args.data_dir),
                output_dir
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    print("\nGenerating summary report...")
    generate_summary_report(output_dir)


if __name__ == "__main__":
    main()
