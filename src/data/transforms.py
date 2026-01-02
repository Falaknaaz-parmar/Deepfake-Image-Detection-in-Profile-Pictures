"""
Data transformation and augmentation utilities
Provides different transforms for training, validation, and testing
"""

from torchvision import transforms
from typing import Dict


def get_transforms(config: Dict, mode: str = 'train'):
    """
    Get appropriate transformations based on mode
    
    Args:
        config (Dict): Configuration dictionary
        mode (str): One of 'train', 'val', or 'test'
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    img_size = config['image']['size']
    normalize_mean = config['image']['normalize_mean']
    normalize_std = config['image']['normalize_std']
    
    if mode == 'train':
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((img_size, img_size)),
        ]
        
        # Add augmentations from config
        aug_config = config.get('augmentation', {})
        
        # Random horizontal flip
        if aug_config.get('random_horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Random rotation
        rotation_degrees = aug_config.get('random_rotation_degrees', 0)
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(rotation_degrees))
        
        # Color jitter
        color_jitter = aug_config.get('color_jitter', {})
        if color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0)
            ))
        
        # Random affine
        affine = aug_config.get('random_affine', {})
        if affine:
            transform_list.append(transforms.RandomAffine(
                degrees=affine.get('degrees', 0),
                translate=tuple(affine.get('translate', [0, 0]))
            ))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ]
    
    return transforms.Compose(transform_list)


def get_inference_transform(config: Dict):
    """
    Get transformations for inference on random images
    
    Args:
        config (Dict): Configuration dictionary
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    return get_transforms(config, mode='test')


def denormalize(tensor, mean, std):
    """
    Denormalize a normalized tensor for visualization
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Denormalized tensor
    """
    mean = tensor.new_tensor(mean).view(-1, 1, 1)
    std = tensor.new_tensor(std).view(-1, 1, 1)
    return tensor * std + mean
