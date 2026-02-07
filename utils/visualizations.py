
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from pathlib import Path
from typing import Optional


# Base output directory structure
OUTPUTS_DIR = Path("outputs")
DATA_DIR = OUTPUTS_DIR / "data"
DRIFT_DIR = OUTPUTS_DIR / "drift"
COMPARISON_DIR = OUTPUTS_DIR / "comparison"
FINAL_DIR = OUTPUTS_DIR / "final"

# Counter for auto-generated filenames
_plot_counter = [0]


def _ensure_output_dirs():
    """Create output directory structure if it doesn't exist."""
    for directory in [DATA_DIR, DRIFT_DIR, COMPARISON_DIR, FINAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _get_filename(category: str, prefix: str, filename: Optional[str] = None) -> Path:
    """
    Generate output filename with proper directory structure.
    
    Args:
        category: Output category (data, drift, comparison, final)
        prefix: Filename prefix for auto-generated names
        filename: Optional explicit filename
    
    Returns:
        Full path to output file
    """
    _ensure_output_dirs()
    
    category_dirs = {
        "data": DATA_DIR,
        "drift": DRIFT_DIR,
        "comparison": COMPARISON_DIR,
        "final": FINAL_DIR,
    }
    
    output_dir = category_dirs.get(category, OUTPUTS_DIR)
    
    if filename is None:
        _plot_counter[0] += 1
        return output_dir / f"{prefix}_{_plot_counter[0]}.jpg"
    
    return output_dir / filename


def viz_2d_data(data: Tensor, filename: Optional[str] = None, category: str = "data"):
    """
    Save 2D scatter plot to organized outputs folder.
    
    Args:
        data: 2D tensor of points [N, 2]
        filename: Optional explicit filename
        category: Output category (data, drift, comparison, final)
    """
    plt.figure()
    data = data.cpu()
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    plt.axis("scaled")
    
    output_path = _get_filename(category, "data_2d", filename)
    plt.savefig(output_path, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_drift_field(
    gen: Tensor,
    pos: Tensor,
    V: Tensor,
    filename: Optional[str] = None,
):
    """
    Quiver plot of drift vectors on generated samples.
    Blue: data (p), Orange: generated (q), Black arrows: drift V.
    
    Saves to outputs/drift/ directory.
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
    
    output_path = _get_filename("drift", "drift", filename)
    plt.savefig(output_path, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_comparison(
    real: Tensor,
    generated: Tensor,
    step: int,
    filename: Optional[str] = None,
):
    """
    Side-by-side scatter of real vs generated samples.
    
    Saves to outputs/comparison/ directory.
    """
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
    
    output_path = _get_filename("comparison", "compare", filename)
    plt.savefig(output_path, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


class Visualizer:
    """
    Real-time matplotlib visualizer for training progress.
    Shows loss curve and 2D data distribution using interactive pyplot.
    """
    
    def __init__(self, title: str = "Training Progress"):
        plt.ion()  # Enable interactive mode for real-time updates
        self.fig, (self.ax_loss, self.ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle(title)
        
        # Loss plot setup
        self.ax_loss.set_title("Training Loss (Log Scale)")
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_yscale("log")
        self.loss_line, = self.ax_loss.plot([], [], label="Loss")
        self.ax_loss.legend()
        self.losses = []
        self.steps = []
        
        # Scatter plot setup
        self.ax_scatter.set_title("Generated vs Target")
        self.ax_scatter.set_aspect("equal")
        self.scat_real = None
        self.scat_gen = None
        
    def update(self, step: int, loss: float, real: Tensor, gen: Tensor):
        """Update plots with new data."""
        # Update loss
        self.steps.append(step)
        self.losses.append(loss)
        self.loss_line.set_data(self.steps, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # Update scatter
        r = real.detach().cpu().numpy()
        g = gen.detach().cpu().numpy()
        
        if self.scat_real is None:
            self.scat_real = self.ax_scatter.scatter(
                r[:, 0], r[:, 1], s=2, alpha=0.3, c="black", label="Target"
            )
            self.scat_gen = self.ax_scatter.scatter(
                g[:, 0], g[:, 1], s=2, alpha=0.3, c="tab:orange", label="Generated"
            )
            self.ax_scatter.legend()
        else:
            self.scat_real.set_offsets(r)
            self.scat_gen.set_offsets(g)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Close the visualizer and keep window open."""
        plt.ioff()
        plt.show()
