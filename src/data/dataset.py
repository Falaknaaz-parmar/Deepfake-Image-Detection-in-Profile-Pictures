"""
Custom Dataset class for Deepfake Detection
Handles loading images and labels from folder structure
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List


class DeepfakeDataset(Dataset):
    """
    Custom Dataset for loading deepfake images
    
    Expected folder structure:
    data_dir/
        ├── Class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── Class2/
            ├── image1.jpg
            └── ...
    
    Args:
        data_dir (str): Path to the data directory
        transform (Callable, optional): Transformations to apply to images
        class_names (List[str], optional): List of class names. If None, will auto-detect
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None
    ):
        self.data_dir = data_dir
        self.transform = transform
        
        # Auto-detect class names from folders if not provided
        if class_names is None:
            self.class_names = sorted([
                d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d))
            ])
        else:
            self.class_names = sorted(class_names)
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from {data_dir}")
        print(f"Classes: {self.class_names}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all image paths and their corresponding labels
        
        Returns:
            List of tuples (image_path, label)
        """
        samples = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            for filename in os.listdir(class_dir):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in supported_extensions:
                    img_path = os.path.join(class_dir, filename)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _get_class_distribution(self) -> dict:
        """Get the distribution of samples per class"""
        distribution = {cls_name: 0 for cls_name in self.class_names}
        
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        
        return distribution
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample by index
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index"""
        return self.class_names[label]
    
    def get_sample_weights(self) -> List[float]:
        """
        Calculate sample weights for balanced sampling
        Useful for handling class imbalance
        
        Returns:
            List of weights for each sample
        """
        # Count samples per class
        class_counts = [0] * len(self.class_names)
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Calculate weights (inverse of class frequency)
        class_weights = [1.0 / count if count > 0 else 0 for count in class_counts]
        
        # Assign weights to each sample
        sample_weights = [class_weights[label] for _, label in self.samples]
        
        return sample_weights
