import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from PIL import Image
import cv2

from config import RESULTS_DIR, DATA_DIR
from dataset import SliceBenchLoader
from models import ModelWrapper, GradCAM
from data_utils import ImageProcessor


def visualize_gradcam_comparison(image_path: str, 
                                model_name: str = "resnet50",
                                output_path: Path = None):
    """Visualize GradCAM for an image."""
    model = ModelWrapper(model_name)
    gradcam = GradCAM(model)
    processor = ImageProcessor()
    
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    tensor = processor.preprocess(image)
    
    cam = gradcam.generate(tensor.unsqueeze(0))
    
    heatmap_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlay = gradcam.overlay_heatmap(heatmap_resized, image_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_colored)
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_accuracy_comparison(results_dir: Path = RESULTS_DIR,
                            output_path: Path = None):
    """Plot accuracy comparison across models and slice types."""
    result_files = list(results_dir.glob("*_results.json"))
    
    if not result_files:
        print("No results found")
        return
    
    data = []
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        model_name = results["model_name"]
        baseline_acc = results["baseline"]["top1_accuracy"]
        
        for slice_result in results["slice_results"]:
            slice_type = slice_result["slice_name"].split("_")[0]
            accuracy = slice_result["top1_accuracy"]
            
            data.append({
                "Model": model_name,
                "Slice Type": slice_type,
                "Accuracy": accuracy,
                "Drop": baseline_acc - accuracy,
            })
    
    if not data:
        print("No data to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    slice_types = df["Slice Type"].unique()
    models = df["Model"].unique()
    
    pivot_acc = df.pivot_table(values="Accuracy", index="Slice Type", columns="Model")
    pivot_acc.plot(kind="bar", ax=ax1)
    ax1.set_title("Accuracy by Slice Type and Model")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_xlabel("Slice Type")
    ax1.legend(title="Model", bbox_to_anchor=(1.05, 1))
    ax1.grid(True, alpha=0.3)
    
    pivot_drop = df.pivot_table(values="Drop", index="Slice Type", columns="Model")
    pivot_drop.plot(kind="bar", ax=ax2)
    ax2.set_title("Accuracy Drop by Slice Type and Model")
    ax2.set_ylabel("Accuracy Drop")
    ax2.set_xlabel("Slice Type")
    ax2.legend(title="Model", bbox_to_anchor=(1.05, 1))
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_bias_summary(results_dir: Path = RESULTS_DIR,
                     output_path: Path = None):
    """Plot summary of detected biases."""
    result_files = list(results_dir.glob("*_results.json"))
    
    if not result_files:
        print("No results found")
        return
    
    bias_data = []
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        model_name = results["model_name"]
        
        for bias in results.get("bias_detections", []):
            bias_data.append({
                "Model": model_name,
                "Bias Type": bias["bias_type"],
                "Has Bias": bias["has_significant_bias"],
                "Severity": bias.get("accuracy_drop", 0),
            })
    
    if not bias_data:
        print("No bias data to plot (models may not show significant biases on synthetic data)")
        return
    
    import pandas as pd
    df = pd.DataFrame(bias_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bias_counts = df[df["Has Bias"]].groupby(["Model", "Bias Type"]).size().unstack(fill_value=0)
    
    if bias_counts.empty:
        ax1.text(0.5, 0.5, "No significant biases detected", ha='center', va='center')
        ax1.set_title("Number of Significant Biases per Model")
    else:
        bias_counts.plot(kind="bar", ax=ax1, stacked=True)
        ax1.set_title("Number of Significant Biases per Model")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Model")
        ax1.legend(title="Bias Type")
        ax1.grid(True, alpha=0.3)
    
    severity = df.groupby(["Model", "Bias Type"])["Severity"].mean().unstack(fill_value=0)
    
    if severity.empty:
        ax2.text(0.5, 0.5, "No biases to measure", ha='center', va='center')
        ax2.set_title("Average Accuracy Drop by Bias Type")
    else:
        severity.plot(kind="bar", ax=ax2)
        ax2.set_title("Average Accuracy Drop by Bias Type")
        ax2.set_ylabel("Accuracy Drop")
        ax2.set_xlabel("Model")
        ax2.legend(title="Bias Type")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_slice_visualization(slice_id: str,
                               data_dir: Path = DATA_DIR / "slicebench",
                               output_path: Path = None):
    """Visualize all variants of a test slice."""
    loader = SliceBenchLoader(data_dir)
    
    slice_info = None
    for s in loader.metadata["slices"]:
        if s["slice_id"] == slice_id:
            slice_info = s
            break
    
    if slice_info is None:
        print(f"Slice {slice_id} not found")
        return
    
    original = Image.open(slice_info["original"])
    
    variants = []
    if "variants" in slice_info:
        for path in slice_info["variants"]:
            variants.append(Image.open(path))
    elif "scale_crops" in slice_info:
        for path in slice_info["scale_crops"][:4]:
            variants.append(Image.open(path))
    elif "blurred" in slice_info:
        variants.append(Image.open(slice_info["blurred"]))
        variants.append(Image.open(slice_info["isolated"]))
    
    n_variants = len(variants)
    n_cols = min(4, n_variants + 1)
    n_rows = (n_variants + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    for idx, variant in enumerate(variants):
        row = (idx + 1) // n_cols
        col = (idx + 1) % n_cols
        axes[row, col].imshow(variant)
        axes[row, col].set_title(f"Variant {idx+1}")
        axes[row, col].axis('off')
    
    for idx in range(n_variants + 1, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Slice: {slice_id} ({slice_info['slice_type']})", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Generating visualizations...")
    
    viz_dir = RESULTS_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating accuracy comparison plot...")
    plot_accuracy_comparison(output_path=viz_dir / "accuracy_comparison.png")
    
    print("Creating bias summary plot...")
    plot_bias_summary(output_path=viz_dir / "bias_summary.png")
    
    print(f"\nVisualizations saved to {viz_dir}")
