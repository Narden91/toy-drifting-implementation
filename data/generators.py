
import torch
import numpy as np
from torch import Tensor

# We can find the device from the calling context or pass it explicitly.
# For simplicity, we'll default to CPU if not provided, but efficient usage requires passing device.

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

def gen_data(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D mixture of 8 Gaussians arranged in a circle."""
    scale = 4.0
    centers = torch.tensor([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=torch.float32, device=device) * scale

    x = 0.5 * torch.randn(n, 2, device=device)
    center_ids = torch.randint(0, 8, (n,), device=device)
    x = (x + centers[center_ids]) / np.sqrt(2)
    return x


def gen_checkerboard(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D checkerboard pattern (4 tiles)."""
    b = torch.randint(0, 2, (n,), device=device)
    i = (torch.randint(0, 2, (n,), device=device) * 2 + b).float()
    j = (torch.randint(0, 2, (n,), device=device) * 2 + b).float()
    u = torch.rand(n, device=device)
    v = torch.rand(n, device=device)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    pts = pts / 2.0
    return pts + 0.05 * torch.randn(n, 2, device=device)
