import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple
import numpy as np


class ModelWrapper:
    """Wrapper for pre-trained models with hook support."""
    
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model(model_name)
        self.model.eval()
        
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def _load_model(self, name: str) -> nn.Module:
        """Load pre-trained model from torchvision."""
        model_fn = getattr(models, name, None)
        if model_fn is None:
            raise ValueError(f"Model {name} not found")
        
        model = model_fn(weights="DEFAULT")
        model = model.to(self.device)
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)
    
    def predict(self, x: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k predictions."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
        return top_probs, top_indices
    
    def get_feature_layer(self) -> nn.Module:
        """Get the last convolutional layer for attribution."""
        if "resnet" in self.model_name:
            return self.model.layer4[-1]
        elif "efficientnet" in self.model_name:
            return self.model.features[-1]
        elif "mobilenet" in self.model_name:
            return self.model.features[-1]
        elif "vit" in self.model_name:
            return self.model.encoder.layers[-1]
        else:
            modules = list(self.model.children())
            return modules[-2] if len(modules) > 1 else modules[-1]
    
    def register_hooks(self):
        """Register forward and backward hooks for gradients."""
        def forward_hook(module, input, output):
            self.activations['value'] = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients['value'] = grad_out[0]
        
        target_layer = self.get_feature_layer()
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        self.remove_hooks()


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.model = model_wrapper
        
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate GradCAM heatmap."""
        self.model.model.eval()
        self.model.register_hooks()
        
        input_tensor = input_tensor.to(self.model.device)
        input_tensor.requires_grad = True
        
        output = self.model.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.model.zero_grad()
        
        class_score = output[0, target_class]
        class_score.backward()
        
        gradients = self.model.gradients['value']
        activations = self.model.activations['value']
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        self.model.remove_hooks()
        
        return cam
    
    def overlay_heatmap(self, heatmap: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on image."""
        from cv2 import resize, applyColorMap, COLORMAP_JET
        
        heatmap_resized = resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = applyColorMap(np.uint8(255 * heatmap_resized), COLORMAP_JET)
        
        overlay = (1 - alpha) * image + alpha * heatmap_colored
        return np.uint8(overlay)


class AttributionAnalyzer:
    """Analyze attribution patterns across image slices."""
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.model = model_wrapper
        self.gradcam = GradCAM(model_wrapper)
    
    def compute_attribution_stats(self, images: List[torch.Tensor], 
                                 target_class: Optional[int] = None) -> Dict[str, float]:
        """Compute attribution statistics for a set of images."""
        heatmaps = []
        
        for img in images:
            cam = self.gradcam.generate(img.unsqueeze(0), target_class)
            heatmaps.append(cam)
        
        heatmaps = np.array(heatmaps)
        
        stats = {
            "mean_activation": heatmaps.mean(),
            "std_activation": heatmaps.std(),
            "max_activation": heatmaps.max(),
            "center_focus": self._compute_center_bias(heatmaps),
            "edge_focus": self._compute_edge_bias(heatmaps),
        }
        
        return stats
    
    def _compute_center_bias(self, heatmaps: np.ndarray) -> float:
        """Measure how much attention is in the center."""
        h, w = heatmaps.shape[1:]
        center_mask = np.zeros((h, w))
        
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
        center_mask[mask] = 1
        
        center_attention = (heatmaps * center_mask).sum(axis=(1, 2))
        total_attention = heatmaps.sum(axis=(1, 2))
        
        return (center_attention / (total_attention + 1e-8)).mean()
    
    def _compute_edge_bias(self, heatmaps: np.ndarray) -> float:
        """Measure how much attention is on edges."""
        h, w = heatmaps.shape[1:]
        edge_width = min(h, w) // 10
        
        edge_mask = np.ones((h, w))
        edge_mask[edge_width:-edge_width, edge_width:-edge_width] = 0
        
        edge_attention = (heatmaps * edge_mask).sum(axis=(1, 2))
        total_attention = heatmaps.sum(axis=(1, 2))
        
        return (edge_attention / (total_attention + 1e-8)).mean()
    
    def compare_attribution_shift(self, original_images: List[torch.Tensor], 
                                 modified_images: List[torch.Tensor]) -> Dict[str, float]:
        """Compare how attribution shifts between original and modified images."""
        orig_stats = self.compute_attribution_stats(original_images)
        mod_stats = self.compute_attribution_stats(modified_images)
        
        shift = {
            f"{key}_shift": mod_stats[key] - orig_stats[key]
            for key in orig_stats.keys()
        }
        
        return shift
