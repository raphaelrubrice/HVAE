"""Sub-module to define diverse utility functions"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List
import sys

# =========================
# Checking utils
# =========================
def has_method(obj: object, method_name: str) -> bool:
    """
    Checks if an object has the desired method
    """
    return callable(getattr(obj, method_name, None))

# =========================
# Training utils
# =========================
def manual_grad_update(model, grad, lr):
    for parameters in model.parameters():
        parameters.data -= lr * grad
    

# =========================
# Synthetic data generation
# =========================

# génération données sur s1 embedées dans r100
def generate_circle_data(n_samples=1000
                         ) -> Tuple[np.ndarray | List]:
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    
    # 3 clusters sur le cercle
    cluster_centers = [0, 2*np.pi/3, 4*np.pi/3]
    labels = np.random.choice(3, n_samples)
    
    for i in range(n_samples):
        angles[i] = cluster_centers[labels[i]] + np.random.randn() * 0.3
    
    # coordonnées 2d
    x = np.cos(angles)
    y = np.sin(angles)
    data_2d = np.stack([x, y], axis=1)
    
    # projection non-linéaire vers r100
    projection = np.random.randn(2, 100)
    data_high = data_2d @ projection
    data_high += np.random.randn(n_samples, 100) * 0.1  # bruit
    
    return data_high, data_2d, labels

def create_circle_training_data(n_samples, 
                                batch_size,
                                return_data_tensor=False, 
                                return_data_2d=False, 
                                return_labels=False
                                ) -> Tuple:
    """
    Creates a synthetic circle dataset and instantiates the dataloader.
    """
    data_high, data_2d, labels = generate_circle_data(n_samples)
    data_tensor = torch.FloatTensor(data_high)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data_tensor, data_2d, labels