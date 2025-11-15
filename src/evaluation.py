import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import json

from .models import ModelWrapper, AttributionAnalyzer
from .data_utils import ImageProcessor


class SliceEvaluator:
    """Evaluate model performance on test slices."""
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.model = model_wrapper
        self.processor = ImageProcessor()
        self.attribution = AttributionAnalyzer(model_wrapper)
        
    def evaluate_accuracy(self, images: List[torch.Tensor], 
                         labels: List[int]) -> Dict[str, float]:
        """Compute accuracy metrics."""
        correct = 0
        top5_correct = 0
        total = len(images)
        
        all_probs = []
        all_preds = []
        
        for img, label in zip(images, labels):
            probs, indices = self.model.predict(img.unsqueeze(0), top_k=5)
            
            pred = indices[0, 0].item()
            all_preds.append(pred)
            all_probs.append(probs[0, 0].item())
            
            if pred == label:
                correct += 1
            
            if label in indices[0].cpu().numpy():
                top5_correct += 1
        
        return {
            "top1_accuracy": correct / total,
            "top5_accuracy": top5_correct / total,
            "mean_confidence": np.mean(all_probs),
            "std_confidence": np.std(all_probs),
        }
    
    def evaluate_calibration(self, images: List[torch.Tensor], 
                           labels: List[int], n_bins: int = 10) -> Dict[str, float]:
        """Compute calibration metrics."""
        all_probs = []
        correct_predictions = []
        
        for img, label in zip(images, labels):
            probs, indices = self.model.predict(img.unsqueeze(0), top_k=1)
            pred_prob = probs[0, 0].item()
            pred_class = indices[0, 0].item()
            
            all_probs.append(pred_prob)
            correct_predictions.append(1 if pred_class == label else 0)
        
        all_probs = np.array(all_probs)
        correct_predictions = np.array(correct_predictions)
        
        brier = brier_score_loss(correct_predictions, all_probs)
        
        try:
            prob_true, prob_pred = calibration_curve(
                correct_predictions, all_probs, n_bins=n_bins, strategy='uniform'
            )
            ece = np.abs(prob_true - prob_pred).mean()
        except:
            ece = 0.0
        
        return {
            "brier_score": brier,
            "expected_calibration_error": ece,
        }
    
    def evaluate_slice(self, images: List[torch.Tensor], 
                      labels: List[int], slice_name: str) -> Dict[str, any]:
        """Full evaluation of a test slice."""
        accuracy_metrics = self.evaluate_accuracy(images, labels)
        calibration_metrics = self.evaluate_calibration(images, labels)
        
        results = {
            "slice_name": slice_name,
            "num_samples": len(images),
            **accuracy_metrics,
            **calibration_metrics,
        }
        
        return results
    
    def compare_slices(self, baseline_images: List[torch.Tensor],
                      baseline_labels: List[int],
                      test_images: List[torch.Tensor],
                      test_labels: List[int],
                      slice_name: str) -> Dict[str, any]:
        """Compare baseline vs test slice."""
        baseline_results = self.evaluate_slice(baseline_images, baseline_labels, "baseline")
        test_results = self.evaluate_slice(test_images, test_labels, slice_name)
        
        accuracy_drop = baseline_results["top1_accuracy"] - test_results["top1_accuracy"]
        confidence_shift = test_results["mean_confidence"] - baseline_results["mean_confidence"]
        
        attribution_shift = self.attribution.compare_attribution_shift(
            baseline_images[:min(10, len(baseline_images))],
            test_images[:min(10, len(test_images))]
        )
        
        return {
            "baseline": baseline_results,
            "test": test_results,
            "accuracy_drop": accuracy_drop,
            "confidence_shift": confidence_shift,
            "attribution_shift": attribution_shift,
        }


class BiasDetector:
    """Detect specific biases in model behavior."""
    
    @staticmethod
    def detect_background_bias(evaluator: SliceEvaluator,
                              original_images: List[torch.Tensor],
                              background_swapped_images: List[torch.Tensor],
                              labels: List[int],
                              threshold: float = 0.1) -> Dict[str, any]:
        """Detect if model relies heavily on background."""
        orig_acc = evaluator.evaluate_accuracy(original_images, labels)["top1_accuracy"]
        swapped_acc = evaluator.evaluate_accuracy(background_swapped_images, labels)["top1_accuracy"]
        
        accuracy_drop = orig_acc - swapped_acc
        
        has_bias = accuracy_drop > threshold
        
        return {
            "bias_type": "background_reliance",
            "original_accuracy": orig_acc,
            "swapped_accuracy": swapped_acc,
            "accuracy_drop": accuracy_drop,
            "has_significant_bias": has_bias,
            "severity": "high" if accuracy_drop > 0.2 else "medium" if accuracy_drop > 0.1 else "low"
        }
    
    @staticmethod
    def detect_position_bias(evaluator: SliceEvaluator,
                           crop_results: Dict[str, Tuple[List[torch.Tensor], List[int]]],
                           threshold: float = 0.15) -> Dict[str, any]:
        """Detect if model is biased toward center or specific positions."""
        position_accuracies = {}
        
        for position, (images, labels) in crop_results.items():
            acc = evaluator.evaluate_accuracy(images, labels)["top1_accuracy"]
            position_accuracies[position] = acc
        
        center_acc = position_accuracies.get("center", 0)
        corner_accs = [v for k, v in position_accuracies.items() if k != "center"]
        
        if corner_accs:
            avg_corner_acc = np.mean(corner_accs)
            position_bias = center_acc - avg_corner_acc
            has_bias = position_bias > threshold
        else:
            position_bias = 0
            has_bias = False
        
        return {
            "bias_type": "position_bias",
            "position_accuracies": position_accuracies,
            "center_advantage": position_bias,
            "has_significant_bias": has_bias,
        }
    
    @staticmethod
    def detect_context_reliance(evaluator: SliceEvaluator,
                               original_images: List[torch.Tensor],
                               context_removed_images: List[torch.Tensor],
                               labels: List[int],
                               threshold: float = 0.1) -> Dict[str, any]:
        """Detect if model relies on context vs object."""
        orig_acc = evaluator.evaluate_accuracy(original_images, labels)["top1_accuracy"]
        no_context_acc = evaluator.evaluate_accuracy(context_removed_images, labels)["top1_accuracy"]
        
        accuracy_drop = orig_acc - no_context_acc
        has_bias = accuracy_drop > threshold
        
        return {
            "bias_type": "context_reliance",
            "original_accuracy": orig_acc,
            "no_context_accuracy": no_context_acc,
            "accuracy_drop": accuracy_drop,
            "has_significant_bias": has_bias,
        }


def save_results(results: Dict, output_path: Path):
    """Save evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
