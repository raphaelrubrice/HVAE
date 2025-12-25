"""Sub-module to define diverse utility functions"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import ive, iv
from typing import Tuple, List
import sys

# =========================
# computation utils
# =========================
class Iv(torch.autograd.Function):
    """
    Differentiable modified Bessel function I_v(x) via SciPy (CPU).
    """
    @staticmethod
    def forward(ctx, order, value):
        ctx.order = order
        ctx.save_for_backward(value)

        x_np = value.detach().cpu().numpy()
        y_np = iv(order, x_np)

        return torch.from_numpy(y_np).to(device=value.device, dtype=value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        order = ctx.order

        x_np = value.detach().cpu().numpy()
        # d/dx I_v(x) = 0.5*(I_{v-1}(x) + I_{v+1}(x))
        di_np = 0.5 * (iv(order - 1, x_np) + iv(order + 1, x_np))
        di = torch.from_numpy(di_np).to(device=value.device, dtype=value.dtype)

        return None, grad_output * di


class Ive(torch.autograd.Function):
    """
    Differentiable scaled modified Bessel ive(v, x) = exp(-x) * I_v(x),
    computed using Iv.apply to reuse its backward.
    """
    @staticmethod
    def forward(ctx, order, value):
        ctx.order = order
        ctx.save_for_backward(value)

        iv_val = Iv.apply(order, value)
        return torch.exp(-value) * iv_val

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        order = ctx.order

        # ive(v,x) = exp(-x) * I_v(x)
        # d/dx ive = exp(-x) * (dI_v/dx - I_v)
        iv_val = Iv.apply(order, value)  # Tensor
        dIv = 0.5 * (Iv.apply(order - 1, value) + Iv.apply(order + 1, value))

        dIve = torch.exp(-value) * (dIv - iv_val)

        return None, grad_output * dIve


# class Ive(torch.autograd.Function):
#     """
#     Computes a differentiable scaled bessel function.
#     """
#     @staticmethod
#     def forward(ctx, order, value):
#         ctx.save_for_backward(value)
#         ctx.order = order
#         ive_val = ive(order, value.detach().cpu().numpy())
#         return torch.from_numpy(ive_val).to(device=value.device, dtype=value.dtype)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         value = ctx.saved_tensors[0]
#         order = ctx.order
#         # derivative from p.14, equation 16:
#         # d/dx ive(order, value) = 1/2 * (ive(order - 1, value) + ive(order + 1, value))
#         di_dval = 0.5 * (ive(order - 1, value.detach().cpu().numpy()) + ive(order + 1, value.detach().cpu().numpy()))
#         di_dval = torch.from_numpy(di_dval).to(device=value.device, dtype=value.dtype)
#         return None, grad_output * di_dval

# =========================
# Checking utils
# =========================
def has_method(obj: object, method_name: str) -> bool:
    """
    Checks if an object has the desired method
    """
    return callable(getattr(obj, method_name, None))

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
                                train_size,
                                ) -> Tuple:
    """
    Creates a synthetic circle dataset and instantiates the dataloader.
    """
    data_high, data_2d, labels = generate_circle_data(n_samples)
    data_high, val_data_high = data_high[:train_size,:], data_high[train_size:,:]
    data_2d, val_data_2d = data_2d[:train_size,:], data_2d[train_size:,:]
    labels, val_labels = labels[:train_size], labels[train_size:]

    # TRAIN
    data_tensor = torch.FloatTensor(data_high)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # VAL
    val_data_tensor = torch.FloatTensor(val_data_high)
    val_dataset = TensorDataset(val_data_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return ((dataloader, data_tensor, data_2d, labels),
            (val_dataloader, val_data_tensor, val_data_2d, val_labels))