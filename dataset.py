# dataset.py

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils import class_weight
import numpy as np
import torch

def get_data_loaders(data_dir, image_size=224, batch_size=16, num_workers=4):
    """
    Creates training, validation, and test data loaders.

    Args:
        data_dir (str): Path to the dataset directory.
        image_size (int): Size to resize the images.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
        list: List of class names.
        torch.Tensor: Class weights for handling imbalance.
    """

    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Paths to dataset splits
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class names
    class_names = train_dataset.classes

    # Compute class weights for handling class imbalance
    labels = [label for _, label in train_dataset]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(labels),
                                                      y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, class_names, class_weights
