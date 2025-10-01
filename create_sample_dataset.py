"""
Create a sample SliceBench dataset from a few test images.
This demonstrates the dataset generation process.
"""

from pathlib import Path
from PIL import Image
import numpy as np

from dataset import SliceBenchGenerator
from config import DATA_DIR


def create_synthetic_backgrounds(output_dir: Path, num_backgrounds: int = 5):
    """Create synthetic background images."""
    bg_dir = output_dir / "backgrounds"
    bg_dir.mkdir(parents=True, exist_ok=True)
    
    bg_paths = []
    
    for i in range(num_backgrounds):
        bg = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        if i == 0:
            bg[:] = [135, 206, 235]
        elif i == 1:
            bg[:] = [34, 139, 34]
        elif i == 2:
            bg[:112, :] = [255, 200, 100]
            bg[112:, :] = [100, 200, 255]
        else:
            noise = np.random.randint(-30, 30, bg.shape, dtype=np.int16)
            base_color = np.random.randint(100, 200, 3)
            bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        bg_img = Image.fromarray(bg)
        bg_path = bg_dir / f"background_{i}.jpg"
        bg_img.save(bg_path)
        bg_paths.append(bg_path)
    
    return bg_paths


def create_sample_images(output_dir: Path, num_samples: int = 10):
    """Create sample test images that look like ImageNet classes."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    sample_paths = []
    labels = []
    class_names = []
    
    import cv2
    
    imagenet_classes = [
        (281, "tabby_cat", [(200, 150, 100), (220, 170, 120), (180, 130, 80)]),
        (207, "golden_retriever", [(220, 190, 140), (240, 210, 160), (200, 170, 120)]),
        (388, "giant_panda", [(240, 240, 240), (30, 30, 30), (200, 200, 200)]),
    ]
    
    for i in range(num_samples):
        class_idx = i % len(imagenet_classes)
        label, class_name, colors = imagenet_classes[class_idx]
        
        img = np.ones((224, 224, 3), dtype=np.uint8)
        
        for y in range(224):
            for x in range(224):
                noise = np.random.randint(-20, 20, 3)
                base_color = colors[np.random.randint(0, len(colors))]
                img[y, x] = np.clip(np.array(base_color) + noise, 0, 255)
        
        center_x, center_y = 112 + np.random.randint(-20, 20), 112 + np.random.randint(-20, 20)
        
        if "cat" in class_name or "panda" in class_name:
            cv2.ellipse(img, (center_x, center_y-20), (40, 35), 0, 0, 360, 
                       tuple(map(int, colors[0])), -1)
            cv2.ellipse(img, (center_x-15, center_y-40), (12, 15), 0, 0, 360,
                       tuple(map(int, colors[0])), -1)
            cv2.ellipse(img, (center_x+15, center_y-40), (12, 15), 0, 0, 360,
                       tuple(map(int, colors[0])), -1)
            cv2.circle(img, (center_x-10, center_y-25), 4, (50, 50, 50), -1)
            cv2.circle(img, (center_x+10, center_y-25), 4, (50, 50, 50), -1)
        else:
            cv2.ellipse(img, (center_x, center_y-10), (45, 40), 0, 0, 360,
                       tuple(map(int, colors[0])), -1)
            cv2.ellipse(img, (center_x, center_y+20), (35, 40), 0, 0, 360,
                       tuple(map(int, colors[0])), -1)
            cv2.ellipse(img, (center_x-25, center_y-35), (15, 12), 0, 0, 360,
                       tuple(map(int, colors[0])), -1)
            cv2.ellipse(img, (center_x+25, center_y-35), (15, 12), 0, 0, 360,
                       tuple(map(int, colors[0])), -1)
        
        sample_img = Image.fromarray(img)
        sample_path = samples_dir / f"sample_{i}_{class_name}.jpg"
        sample_img.save(sample_path)
        
        sample_paths.append(sample_path)
        labels.append(label)
        class_names.append(class_name)
    
    return sample_paths, labels, class_names


def generate_sample_dataset():
    """Generate a complete sample SliceBench dataset."""
    print("Creating sample SliceBench dataset...")
    
    output_dir = DATA_DIR / "sample_source"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating synthetic backgrounds...")
    backgrounds = create_synthetic_backgrounds(output_dir)
    
    print("Creating sample images...")
    sample_paths, labels, class_names = create_sample_images(output_dir)
    
    print("Generating test slices...")
    generator = SliceBenchGenerator()
    
    for i, (img_path, label, class_name) in enumerate(zip(sample_paths, labels, class_names)):
        slice_id = f"{class_name}_{i}"
        
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
    print(f"\nSample dataset generated successfully!")
    print(f"Location: {DATA_DIR / 'slicebench'}")
    print(f"Total slices: {len(generator.metadata['slices'])}")


if __name__ == "__main__":
    generate_sample_dataset()
