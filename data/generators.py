
import torch
import numpy as np
from torch import Tensor
from typing import Optional


def get_device(config: Optional[dict] = None) -> torch.device:
    """
    Get the device to use for computation.
    
    Args:
        config: Optional configuration dict with 'performance' -> 'device' setting
    
    Returns:
        torch.device to use
    """
    # Check if config specifies a device
    if config is not None and 'performance' in config:
        device_str = config['performance'].get('device', 'auto')
        if device_str != 'auto':
            return torch.device(device_str)
    
    # Auto-detect the best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# Cache centers to avoid repeated tensor creation
_CENTERS_CACHE: dict[torch.device, Tensor] = {}
_SQRT2_INV = 1.0 / np.sqrt(2)


def _get_8gaussian_centers(device: torch.device) -> Tensor:
    """Get cached 8-Gaussian centers for the given device."""
    if device not in _CENTERS_CACHE:
        scale = 4.0
        _CENTERS_CACHE[device] = torch.tensor([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [_SQRT2_INV, _SQRT2_INV],
            [_SQRT2_INV, -_SQRT2_INV],
            [-_SQRT2_INV, _SQRT2_INV],
            [-_SQRT2_INV, -_SQRT2_INV]
        ], dtype=torch.float32, device=device) * scale
    return _CENTERS_CACHE[device]


def gen_data(n: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Generate 2D mixture of 8 Gaussians arranged in a circle.
    
    Optimized with cached centers and efficient tensor operations.
    
    Args:
        n: Number of samples to generate
        device: Target device (defaults to auto-detected device)
    
    Returns:
        Tensor of shape [n, 2] with sampled points
    """
    if device is None:
        device = DEVICE
    
    centers = _get_8gaussian_centers(device)
    
    # Pre-allocate and generate noise
    noise = torch.randn(n, 2, device=device) * 0.5
    
    # Random center selection
    center_ids = torch.randint(0, 8, (n,), device=device)
    
    # Combine noise with selected centers
    samples = (noise + centers[center_ids]) * _SQRT2_INV
    
    return samples


def gen_checkerboard(n: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Generate 2D checkerboard pattern (4 tiles).
    
    Optimized with fused operations for better performance.
    
    Args:
        n: Number of samples to generate
        device: Target device (defaults to auto-detected device)
    
    Returns:
        Tensor of shape [n, 2] with sampled points
    """
    if device is None:
        device = DEVICE
    
    # Generate all random values in minimal calls
    b = torch.randint(0, 2, (n,), device=device, dtype=torch.float32)
    
    # Use randint directly with dtype=float32 to avoid conversion
    i_base = torch.randint(0, 2, (n,), device=device, dtype=torch.float32)
    j_base = torch.randint(0, 2, (n,), device=device, dtype=torch.float32)
    
    # Compute grid positions
    i = i_base * 2.0 + b
    j = j_base * 2.0 + b
    
    # Generate uniform offsets
    offsets = torch.rand(n, 2, device=device)
    
    # Stack coordinates
    pts = torch.stack([i + offsets[:, 0], j + offsets[:, 1]], dim=1)
    
    # Center and scale
    pts = (pts - 2.0) * 0.5
    
    # Add noise
    noise = torch.randn(n, 2, device=device) * 0.05
    
    return pts + noise
