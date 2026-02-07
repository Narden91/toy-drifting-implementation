
import torch
import matplotlib.pyplot as plt
from torch import Tensor
import os

_plot_counter = [0]


def _get_filename(prefix: str, filename: str | None) -> str:
    # Ensure directory exists for output if needed, or just save to current dir
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
