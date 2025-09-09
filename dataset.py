import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

from data_utils import (
    ImageProcessor, BackgroundSwapper, CropGenerator, 
    OverlayGenerator, ContextRemover
)
from config import DATA_DIR


class SliceBenchGenerator:
    """Generate SliceBench test slices from source images."""
    
    def __init__(self, output_dir: Path = DATA_DIR / "slicebench"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = ImageProcessor()
        self.bg_swapper = BackgroundSwapper()
        self.crop_gen = CropGenerator()
        self.overlay_gen = OverlayGenerator()
        self.context_remover = ContextRemover()
        
        self.metadata = {
            "slices": [],
            "source_images": [],
        }
    
    def add_source_image(self, image_path: Path, label: int, class_name: str):
        """Add a source image to the dataset."""
        self.metadata["source_images"].append({
            "path": str(image_path),
            "label": label,
            "class_name": class_name,
        })
    
    def generate_background_swap_slice(self, 
                                       image_path: Path, 
                                       label: int,
                                       backgrounds: List[Path],
                                       slice_id: str) -> Dict:
        """Generate background swap test slice."""
        image = Image.open(image_path).convert('RGB')
        
        slice_dir = self.output_dir / "background_swap" / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = slice_dir / "original.jpg"
        image.save(original_path)
        
        swapped_images = []
        for i, bg_path in enumerate(backgrounds):
            bg = Image.open(bg_path).convert('RGB')
            swapped = self.bg_swapper.swap_background(image, bg)
            
            save_path = slice_dir / f"bg_swap_{i}.jpg"
            swapped.save(save_path)
            swapped_images.append(str(save_path))
        
        slice_info = {
            "slice_id": slice_id,
            "slice_type": "background_swap",
            "original": str(original_path),
            "variants": swapped_images,
            "label": label,
        }
        
        self.metadata["slices"].append(slice_info)
        return slice_info
    
    def generate_crop_slice(self, 
                           image_path: Path, 
                           label: int,
                           slice_id: str) -> Dict:
        """Generate crop variation test slice."""
        image = Image.open(image_path).convert('RGB')
        
        slice_dir = self.output_dir / "crop_variations" / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = slice_dir / "original.jpg"
        image.save(original_path)
        
        crops = self.crop_gen.generate_crops(image)
        crop_paths = []
        for i, crop in enumerate(crops):
            save_path = slice_dir / f"crop_{i}.jpg"
            crop.save(save_path)
            crop_paths.append(str(save_path))
        
        corner_crops = self.crop_gen.generate_corner_crops(image)
        corner_paths = {}
        for position, crop in corner_crops:
            save_path = slice_dir / f"crop_{position}.jpg"
            crop.save(save_path)
            corner_paths[position] = str(save_path)
        
        slice_info = {
            "slice_id": slice_id,
            "slice_type": "crop_variations",
            "original": str(original_path),
            "scale_crops": crop_paths,
            "position_crops": corner_paths,
            "label": label,
        }
        
        self.metadata["slices"].append(slice_info)
        return slice_info
    
    def generate_context_removal_slice(self,
                                       image_path: Path,
                                       label: int,
                                       slice_id: str) -> Dict:
        """Generate context removal test slice."""
        image = Image.open(image_path).convert('RGB')
        
        slice_dir = self.output_dir / "context_removal" / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = slice_dir / "original.jpg"
        image.save(original_path)
        
        blurred = self.context_remover.blur_background(image)
        blurred_path = slice_dir / "blurred_bg.jpg"
        blurred.save(blurred_path)
        
        isolated = self.context_remover.isolate_object(image)
        isolated_path = slice_dir / "isolated.jpg"
        isolated.save(isolated_path)
        
        slice_info = {
            "slice_id": slice_id,
            "slice_type": "context_removal",
            "original": str(original_path),
            "blurred": str(blurred_path),
            "isolated": str(isolated_path),
            "label": label,
        }
        
        self.metadata["slices"].append(slice_info)
        return slice_info
    
    def generate_watermark_slice(self,
                                image_path: Path,
                                label: int,
                                slice_id: str) -> Dict:
        """Generate watermark test slice."""
        image = Image.open(image_path).convert('RGB')
        
        slice_dir = self.output_dir / "watermark_test" / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = slice_dir / "original.jpg"
        image.save(original_path)
        
        watermarks = []
        for position in ["bottom_right", "top_left", "top_right", "bottom_left"]:
            marked = self.overlay_gen.add_watermark(image, "SAMPLE", position)
            save_path = slice_dir / f"watermark_{position}.jpg"
            marked.save(save_path)
            watermarks.append(str(save_path))
        
        slice_info = {
            "slice_id": slice_id,
            "slice_type": "watermark_test",
            "original": str(original_path),
            "variants": watermarks,
            "label": label,
        }
        
        self.metadata["slices"].append(slice_info)
        return slice_info
    
    def generate_texture_bias_slice(self,
                                    image_path: Path,
                                    label: int,
                                    slice_id: str) -> Dict:
        """Generate texture/color bias test slice."""
        image = Image.open(image_path).convert('RGB')
        
        slice_dir = self.output_dir / "texture_bias" / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = slice_dir / "original.jpg"
        image.save(original_path)
        
        colors = [
            ("red", (255, 0, 0)),
            ("green", (0, 255, 0)),
            ("blue", (0, 0, 255)),
            ("yellow", (255, 255, 0)),
        ]
        
        patches = []
        for color_name, color_rgb in colors:
            patched = self.overlay_gen.add_colored_patch(image, color_rgb, size=60)
            save_path = slice_dir / f"patch_{color_name}.jpg"
            patched.save(save_path)
            patches.append(str(save_path))
        
        slice_info = {
            "slice_id": slice_id,
            "slice_type": "texture_bias",
            "original": str(original_path),
            "variants": patches,
            "label": label,
        }
        
        self.metadata["slices"].append(slice_info)
        return slice_info
    
    def save_metadata(self):
        """Save dataset metadata."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")


class SliceBenchLoader:
    """Load SliceBench dataset for evaluation."""
    
    def __init__(self, data_dir: Path = DATA_DIR / "slicebench"):
        self.data_dir = data_dir
        
        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.processor = ImageProcessor()
    
    def load_slice(self, slice_id: str) -> Tuple[List[torch.Tensor], List[int], Dict]:
        """Load a specific test slice."""
        slice_info = None
        for s in self.metadata["slices"]:
            if s["slice_id"] == slice_id:
                slice_info = s
                break
        
        if slice_info is None:
            raise ValueError(f"Slice {slice_id} not found")
        
        images = []
        labels = []
        
        slice_type = slice_info["slice_type"]
        
        if "variants" in slice_info:
            for path in slice_info["variants"]:
                img = Image.open(path).convert('RGB')
                tensor = self.processor.preprocess(img)
                images.append(tensor)
                labels.append(slice_info["label"])
        
        elif "scale_crops" in slice_info:
            for path in slice_info["scale_crops"]:
                img = Image.open(path).convert('RGB')
                tensor = self.processor.preprocess(img)
                images.append(tensor)
                labels.append(slice_info["label"])
        
        elif "blurred" in slice_info:
            for key in ["blurred", "isolated"]:
                img = Image.open(slice_info[key]).convert('RGB')
                tensor = self.processor.preprocess(img)
                images.append(tensor)
                labels.append(slice_info["label"])
        
        return images, labels, slice_info
    
    def load_all_slices(self, slice_type: str = None) -> List[Tuple[List[torch.Tensor], List[int], Dict]]:
        """Load all slices, optionally filtered by type."""
        results = []
        
        for slice_info in self.metadata["slices"]:
            if slice_type and slice_info["slice_type"] != slice_type:
                continue
            
            slice_id = slice_info["slice_id"]
            images, labels, info = self.load_slice(slice_id)
            results.append((images, labels, info))
        
        return results
    
    def get_baseline_images(self, slice_ids: List[str] = None) -> Tuple[List[torch.Tensor], List[int]]:
        """Get original/baseline images for comparison."""
        images = []
        labels = []
        
        for slice_info in self.metadata["slices"]:
            if slice_ids and slice_info["slice_id"] not in slice_ids:
                continue
            
            original_path = slice_info["original"]
            img = Image.open(original_path).convert('RGB')
            tensor = self.processor.preprocess(img)
            images.append(tensor)
            labels.append(slice_info["label"])
        
        return images, labels
