import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import cv2

from config import IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMAGE_SIZE


class ImageProcessor:
    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE):
        self.image_size = image_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.to_tensor = T.ToTensor()
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = self.to_tensor(img)
        return self.normalize(tensor)
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = tensor * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (img * 255).astype(np.uint8)


class BackgroundSwapper:
    """Swap image backgrounds to test background bias."""
    
    def __init__(self, background_dir: Optional[Path] = None):
        self.backgrounds = []
        if background_dir and background_dir.exists():
            self.backgrounds = list(background_dir.glob("*.jpg")) + list(background_dir.glob("*.png"))
    
    def create_mask(self, image: Image.Image, method: str = "grabcut") -> np.ndarray:
        """Simple foreground extraction using GrabCut or saliency."""
        img_np = np.array(image)
        mask = np.zeros(img_np.shape[:2], np.uint8)
        
        if method == "grabcut":
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            h, w = img_np.shape[:2]
            rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
            
            try:
                cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            except:
                mask = np.ones(img_np.shape[:2], np.uint8)
        
        return mask
    
    def swap_background(self, image: Image.Image, background: Image.Image) -> Image.Image:
        """Replace background while keeping foreground."""
        w, h = image.size
        background = background.resize((w, h), Image.BILINEAR)
        
        mask = self.create_mask(image)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
        
        result = Image.composite(image, background, mask_pil)
        return result
    
    def generate_variations(self, image: Image.Image, num_variations: int = 3) -> List[Image.Image]:
        """Generate multiple background variations."""
        if not self.backgrounds:
            return [image]
        
        variations = []
        for i in range(min(num_variations, len(self.backgrounds))):
            bg = Image.open(self.backgrounds[i]).convert('RGB')
            variations.append(self.swap_background(image, bg))
        
        return variations


class CropGenerator:
    """Generate crop variations to test object localization bias."""
    
    @staticmethod
    def generate_crops(image: Image.Image, num_crops: int = 5) -> List[Image.Image]:
        """Generate random crops of varying sizes."""
        w, h = image.size
        crops = []
        
        for scale in [0.5, 0.6, 0.7, 0.8, 0.9]:
            crop_w = int(w * scale)
            crop_h = int(h * scale)
            
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            
            crop = image.crop((left, top, left + crop_w, top + crop_h))
            crops.append(crop.resize((w, h), Image.BILINEAR))
        
        return crops[:num_crops]
    
    @staticmethod
    def generate_corner_crops(image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """Generate crops from different corners to test position bias."""
        w, h = image.size
        crop_size = min(w, h) // 2
        
        positions = {
            "center": ((w-crop_size)//2, (h-crop_size)//2),
            "top_left": (0, 0),
            "top_right": (w-crop_size, 0),
            "bottom_left": (0, h-crop_size),
            "bottom_right": (w-crop_size, h-crop_size),
        }
        
        crops = []
        for name, (x, y) in positions.items():
            crop = image.crop((x, y, x+crop_size, y+crop_size))
            crops.append((name, crop.resize((w, h), Image.BILINEAR)))
        
        return crops


class OverlayGenerator:
    """Add watermarks and overlays to test spurious correlation."""
    
    @staticmethod
    def add_watermark(image: Image.Image, text: str = "TEST", position: str = "bottom_right") -> Image.Image:
        """Add text watermark to image."""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        positions = {
            "bottom_right": (w - 100, h - 30),
            "bottom_left": (10, h - 30),
            "top_right": (w - 100, 30),
            "top_left": (10, 30),
        }
        
        pos = positions.get(position, positions["bottom_right"])
        cv2.putText(img_np, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return Image.fromarray(img_np)
    
    @staticmethod
    def add_colored_patch(image: Image.Image, color: Tuple[int, int, int] = (255, 0, 0), 
                         size: int = 50, position: str = "top_left") -> Image.Image:
        """Add colored patch to test texture/color bias."""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        positions = {
            "top_left": (0, 0),
            "top_right": (w - size, 0),
            "bottom_left": (0, h - size),
            "bottom_right": (w - size, h - size),
        }
        
        x, y = positions.get(position, positions["top_left"])
        img_np[y:y+size, x:x+size] = color
        
        return Image.fromarray(img_np)


class ContextRemover:
    """Remove context to test object-vs-context reliance."""
    
    @staticmethod
    def blur_background(image: Image.Image, blur_strength: int = 51) -> Image.Image:
        """Blur background while keeping foreground sharp."""
        img_np = np.array(image)
        
        mask = np.zeros(img_np.shape[:2], np.uint8)
        h, w = img_np.shape[:2]
        rect = (int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6))
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        except:
            fg_mask = np.ones(img_np.shape[:2], np.uint8)
        
        blurred = cv2.GaussianBlur(img_np, (blur_strength, blur_strength), 0)
        
        result = img_np * fg_mask[:, :, np.newaxis] + blurred * (1 - fg_mask[:, :, np.newaxis])
        
        return Image.fromarray(result.astype(np.uint8))
    
    @staticmethod
    def isolate_object(image: Image.Image, background_color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
        """Isolate object on uniform background."""
        img_np = np.array(image)
        
        mask = np.zeros(img_np.shape[:2], np.uint8)
        h, w = img_np.shape[:2]
        rect = (int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6))
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_np, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        except:
            fg_mask = np.ones(img_np.shape[:2], np.uint8)
        
        bg = np.full_like(img_np, background_color)
        result = img_np * fg_mask[:, :, np.newaxis] + bg * (1 - fg_mask[:, :, np.newaxis])
        
        return Image.fromarray(result.astype(np.uint8))
