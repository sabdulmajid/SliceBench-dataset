"""Unit tests for data_utils module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
from PIL import Image
from src.data_utils import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Test ImageProcessor functionality."""
    
    def setUp(self):
        """Create test image."""
        self.processor = ImageProcessor()
        self.test_image = Image.new('RGB', (224, 224), color='red')
    
    def test_preprocess_shape(self):
        """Test preprocessing returns correct shape."""
        tensor = self.processor.preprocess(self.test_image)
        self.assertEqual(tensor.shape, (3, 224, 224))
    
    def test_preprocess_normalization(self):
        """Test preprocessing normalizes values."""
        tensor = self.processor.preprocess(self.test_image)
        # Values should be normalized (not in 0-255 range)
        self.assertTrue(tensor.min() < 0 or tensor.max() < 1)


if __name__ == '__main__':
    unittest.main()
