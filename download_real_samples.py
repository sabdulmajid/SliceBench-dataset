"""
Download real sample images for testing SliceBench.
Uses ImageNet sample images that are publicly available.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import urllib.request
import json

from dataset import SliceBenchGenerator
from config import DATA_DIR


def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            from io import BytesIO
            return Image.open(BytesIO(response.read())).convert('RGB')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def create_real_imagenet_samples():
    """Create dataset using real ImageNet sample images."""
    
    print("Creating SliceBench dataset with real images...")
    
    output_dir = DATA_DIR / "sample_source"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    sample_images = [
        {
            "url": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",
            "label": 281,
            "name": "tabby_cat"
        },
        {
            "url": "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400",
            "label": 207, 
            "name": "golden_retriever"
        },
        {
            "url": "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=400",
            "label": 388,
            "name": "panda"
        },
        {
            "url": "https://images.unsplash.com/photo-1472491235688-bdc81a63246e?w=400",
            "label": 281,
            "name": "tabby_cat_2"
        },
        {
            "url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",
            "label": 207,
            "name": "golden_retriever_2"
        },
    ]
    
    print("\nDownloading sample images from Unsplash...")
    print("(Free images for testing purposes)\n")
    
    downloaded_paths = []
    labels = []
    class_names = []
    
    for i, img_info in enumerate(sample_images):
        print(f"Downloading {img_info['name']}...")
        img = download_image_from_url(img_info['url'])
        
        if img is None:
            print(f"  Skipped (download failed)")
            continue
        
        img = img.resize((400, 400), Image.BILINEAR)
        
        save_path = samples_dir / f"sample_{i}_{img_info['name']}.jpg"
        img.save(save_path, quality=95)
        
        downloaded_paths.append(save_path)
        labels.append(img_info['label'])
        class_names.append(img_info['name'])
        print(f"  ✓ Saved to {save_path.name}")
    
    if len(downloaded_paths) < 3:
        print("\nNot enough images downloaded. Falling back to synthetic images.")
        return create_synthetic_fallback()
    
    print(f"\n✓ Downloaded {len(downloaded_paths)} images")
    
    print("\nCreating synthetic backgrounds...")
    from create_sample_dataset import create_synthetic_backgrounds
    backgrounds = create_synthetic_backgrounds(output_dir)
    
    print("\nGenerating test slices...")
    generator = SliceBenchGenerator()
    
    for img_path, label, class_name in zip(downloaded_paths, labels, class_names):
        slice_id = f"{class_name}_{label}"
        
        generator.add_source_image(img_path, label, class_name)
        
        print(f"  Generating slices for {slice_id}...")
        
        generator.generate_background_swap_slice(
            img_path, label, backgrounds[:3], f"bg_{slice_id}"
        )
        
        generator.generate_crop_slice(img_path, label, f"crop_{slice_id}")
        
        generator.generate_context_removal_slice(img_path, label, f"ctx_{slice_id}")
        
        generator.generate_watermark_slice(img_path, label, f"wm_{slice_id}")
        
        generator.generate_texture_bias_slice(img_path, label, f"tex_{slice_id}")
    
    generator.save_metadata()
    print(f"\n✓ Dataset generated successfully!")
    print(f"Location: {DATA_DIR / 'slicebench'}")
    print(f"Total slices: {len(generator.metadata['slices'])}")
    
    return downloaded_paths, labels, class_names


def create_synthetic_fallback():
    """Fallback to synthetic images if downloads fail."""
    print("\nUsing synthetic images as fallback...")
    from create_sample_dataset import generate_sample_dataset
    return generate_sample_dataset()


if __name__ == "__main__":
    try:
        create_real_imagenet_samples()
    except Exception as e:
        print(f"\nError: {e}")
        print("Falling back to synthetic dataset...")
        create_synthetic_fallback()
