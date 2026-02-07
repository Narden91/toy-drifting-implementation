"""
Drifting Models from Scratch
============================

A minimal implementation of Drifting Models for 2D toy data.
Unlike diffusion/flow models that iterate at inference, drifting models evolve the
pushforward distribution during training and generate in a single forward pass (1-NFE).
The drifting field V governs sample movement: V -> 0 as generated matches data.

Reference: Deng et al., "Generative Modeling via Drifting", ICML 2026
"""

from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# Data Generation
# =============================================================================

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

# =============================================================================
# Neural Network
# =============================================================================

class Net(nn.Module):
    """MLP generator: noise -> 2D samples. SELU activations, 4 hidden layers."""

    def __init__(self, noise_dim: int = 32, hidden_dim: int = 256, out_dim: int = 2):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)

# =============================================================================
# Drifting Field: Compute V  (Algorithm 2 from paper)
# =============================================================================
#
# Mean-shift drifting field (compact form):
#
#   V(x) = (1 / (Z_p Z_q)) E_{y+~p, y-~q}[ k(x,y+) k(x,y-) (y+ - y-) ]
#
# where k(x,y) = exp(-||x-y|| / tau).
#
# Implementation uses doubly-normalized softmax (over both x and y axes)
# and a factorized weight computation to avoid O(N * N_pos * N_neg) cost.
#

def compute_drift(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
) -> Tensor:
    """
    Compute the mean-shift drifting field V.

    Args:
        x:     Query points (generated samples) [N, D]
        y_pos: Positive samples (data) [N_pos, D]
        y_neg: Negative samples (generated) [N_neg, D]
        temp:  Kernel temperature

    Returns:
        V: Drift vectors [N, D]
    """
    N, N_pos = x.shape[0], y_pos.shape[0]

    # pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos)                   # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)                    # [N, N_neg]

    # mask self-interactions (y_neg = x in standard usage)
    if N == y_neg.shape[0]:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # logits: -dist / temp
    logit = torch.cat([
        -dist_pos / temp,
        -dist_neg / temp,
    ], dim=1)                                           # [N, N_pos + N_neg]

    # doubly-normalized affinity: softmax over both dims, geometric mean
    A_row = logit.softmax(dim=-1)                       # normalize over y
    A_col = logit.softmax(dim=-2)                       # normalize over x
    A = (A_row * A_col).sqrt()

    # split into positive and negative affinities
    A_pos = A[:, :N_pos]                                # [N, N_pos]
    A_neg = A[:, N_pos:]                                # [N, N_neg]

    # factorized weights for compact form:
    # W_pos[i,j] = A_pos[i,j] * sum_k A_neg[i,k]
    # W_neg[i,k] = A_neg[i,k] * sum_j A_pos[i,j]
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    # V[i] = sum_j sum_k A_pos[i,j] A_neg[i,k] (y_pos[j] - y_neg[k])
    V = W_pos @ y_pos - W_neg @ y_neg

    return V

# =============================================================================
# Drifting Loss  (Algorithm 1 from paper)
# =============================================================================
#
# L = E[ || f(eps) - stopgrad(f(eps) + V(f(eps))) ||^2 ]
#
# The loss value equals ||V||^2. Gradients flow only through the prediction
# f(eps), not through the frozen target f(eps) + V.
#

def drifting_loss(gen: Tensor, pos: Tensor, temp: float = 0.05) -> Tensor:
    """
    Compute per-sample drifting loss.

    Args:
        gen: Generated samples [N, D] (with gradient)
        pos: Data (positive) samples [N_pos, D]
        temp: Kernel temperature

    Returns:
        Per-sample loss [N] (equals ||V(x)||^2 per sample)
    """
    with torch.no_grad():
        V = compute_drift(gen, pos, gen, temp=temp)
        target = (gen + V).detach()
    return ((gen - target) ** 2).sum(dim=-1)

# =============================================================================
# Drifting Model
# =============================================================================

class DriftingModel(nn.Module):
    """
    Drifting Model: one-step generator trained via training-time distribution evolution.

    The generator maps noise to samples. The drifting field V provides a training
    signal: it points each generated sample toward where it should move to better
    match the data distribution. When V -> 0, distributions match (equilibrium).
    """

    def __init__(self, noise_dim: int = 32, hidden_dim: int = 256, data_dim: int = 2,
                 temp: float = 0.05):
        super().__init__()
        self.net = Net(noise_dim, hidden_dim, data_dim)
        self.noise_dim = noise_dim
        self.temp = temp

    def forward(self, pos: Tensor, n_gen: int | None = None) -> Tensor:
        """
        Compute per-sample drifting loss.

        Args:
            pos: Data samples [N_pos, D]
            n_gen: Number of generated samples (defaults to N_pos)

        Returns:
            Per-sample loss [n_gen]
        """
        n = n_gen or pos.shape[0]
        z = torch.randn(n, self.noise_dim, device=pos.device)
        gen = self.net(z)
        return drifting_loss(gen, pos, temp=self.temp)

    @torch.no_grad()
    def generate(self, n: int) -> Tensor:
        """Generate n samples (1-NFE)."""
        z = torch.randn(n, self.noise_dim, device=next(self.parameters()).device)
        return self.net(z)

# =============================================================================
# Visualization
# =============================================================================

_plot_counter = [0]


def _get_filename(prefix: str, filename: str | None) -> str:
    if filename is None:
        _plot_counter[0] += 1
        return f"{prefix}_{_plot_counter[0]}.jpg"
    return filename


def viz_2d_data(data: Tensor, filename: str | None = None):
    """Save 2D scatter plot."""
    plt.figure()
    data = data.cpu()
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    plt.axis("scaled")
    plt.savefig(_get_filename("data_2d", filename), format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_drift_field(
    gen: Tensor,
    pos: Tensor,
    V: Tensor,
    filename: str | None = None,
):
    """
    Quiver plot of drift vectors on generated samples.
    Blue: data (p), Orange: generated (q), Black arrows: drift V.
    """
    g = gen.detach().cpu().numpy()
    p = pos.detach().cpu().numpy()
    v = V.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(p[:, 0], p[:, 1], s=3, alpha=0.2, c="tab:blue", label="data (p)")
    plt.scatter(g[:, 0], g[:, 1], s=20, c="tab:orange", label="generated (q)")
    plt.quiver(
        g[:, 0], g[:, 1], v[:, 0], v[:, 1],
        scale=3, color="black", alpha=0.7, width=0.004,
    )
    plt.legend(fontsize=8)
    plt.axis("scaled")
    plt.grid(True, alpha=0.2)
    plt.savefig(_get_filename("drift", filename), format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_comparison(
    real: Tensor,
    generated: Tensor,
    step: int,
    filename: str | None = None,
):
    """Side-by-side scatter of real vs generated samples."""
    r = real.detach().cpu().numpy()
    g = generated.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.scatter(r[:, 0], r[:, 1], s=2, alpha=0.3, c="black")
    ax1.set_title("Target (p)")
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax2.scatter(g[:, 0], g[:, 1], s=2, alpha=0.3, c="tab:orange")
    ax2.set_title(f"Generated (step {step})")
    ax2.set_aspect("equal")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(_get_filename("compare", filename), format="jpg", dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Training
# =============================================================================

def train(
    model: DriftingModel,
    data_fn: Callable[[int], Tensor],
    n_iter: int = 5000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    sample_every: int = 1000,
    n_samples: int = 4096,
):
    """Train a drifting model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_iter)

    for i in pbar:
        pos = data_fn(batch_size)

        loss = model(pos, n_gen=batch_size).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            pbar.set_description(f"loss: {loss.item():.4f}")

        if (i + 1) % sample_every == 0:
            model.eval()

            # sample comparison
            gen_vis = model.generate(n_samples)
            real_vis = data_fn(n_samples)
            viz_comparison(real_vis, gen_vis, i + 1)

            # drift field visualization
            gen_drift = model.generate(200)
            pos_drift = data_fn(2000)
            V = compute_drift(gen_drift, pos_drift, gen_drift, temp=model.temp)
            viz_drift_field(gen_drift, pos_drift, V)

            model.train()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # visualize data
    viz_2d_data(gen_data(8096), filename="data_8gaussians.jpg")
    viz_2d_data(gen_checkerboard(8096), filename="data_checkerboard.jpg")

    # visualize drift field before training: random points drifting toward data
    torch.manual_seed(42)
    gen_init = torch.randn(150, 2, device=DEVICE) * 2.0
    pos_init = gen_data(2000)
    with torch.no_grad():
        V_init = compute_drift(gen_init, pos_init, gen_init, temp=0.2)
    viz_drift_field(gen_init, pos_init, V_init, filename="drift_initial.jpg")

    # train on 8 Gaussians
    print("\n--- Training on 8 Gaussians ---")
    model = DriftingModel(noise_dim=32, hidden_dim=256, temp=0.05).to(DEVICE)
    train(model, gen_data, n_iter=5000, batch_size=2048)

    # final samples + drift
    model.eval()
    gen_final = model.generate(4096)
    viz_2d_data(gen_final, filename="final_8gaussians.jpg")

    gen_drift = model.generate(200)
    pos_drift = gen_data(2000)
    with torch.no_grad():
        V_final = compute_drift(gen_drift, pos_drift, gen_drift, temp=0.05)
    viz_drift_field(gen_drift, pos_drift, V_final, filename="drift_final_8gaussians.jpg")

    # train on checkerboard
    print("\n--- Training on Checkerboard ---")
    model2 = DriftingModel(noise_dim=32, hidden_dim=256, temp=0.05).to(DEVICE)
    train(model2, gen_checkerboard, n_iter=5000, batch_size=2048)

    model2.eval()
    gen_final2 = model2.generate(4096)
    viz_2d_data(gen_final2, filename="final_checkerboard.jpg")

    gen_drift2 = model2.generate(200)
    pos_drift2 = gen_checkerboard(2000)
    with torch.no_grad():
        V_final2 = compute_drift(gen_drift2, pos_drift2, gen_drift2, temp=0.05)
    viz_drift_field(gen_drift2, pos_drift2, V_final2, filename="drift_final_checkerboard.jpg")

    print("Done.")