
import torch
from tqdm import trange
from typing import Callable, Optional
from torch import Tensor
from models.drifting_model import DriftingModel
from core.drift import compute_drift
from utils.visualizations import viz_comparison, viz_drift_field

# Import real-time visualizer
try:
    from utils.realtime_viz import RealtimeVisualizer
    REALTIME_VIZ_AVAILABLE = True
except ImportError:
    REALTIME_VIZ_AVAILABLE = False
    print("Warning: PyQtGraph not available. Real-time visualization disabled.")
    print("Install with: pip install pyqtgraph PyQt6")


def train(
    model: DriftingModel,
    data_fn: Callable[[int], Tensor],
    config: dict,
):
    """Train a drifting model using configuration dictionary."""
    n_iter = config['n_iter']
    batch_size = config['batch_size']
    lr = config['lr']
    sample_every = config['sample_every']
    n_samples = config['n_samples']
    
    # Visualization config
    viz_config = config.get('visualization', {})
    viz_enabled = viz_config.get('enabled', True) and REALTIME_VIZ_AVAILABLE
    viz_target_fps = viz_config.get('target_fps', 120)
    viz_update_every = viz_config.get('update_every', 1)
    viz_max_points = viz_config.get('max_display_points', 2000)
    viz_history_size = viz_config.get('loss_history_size', 1000)
    
    # Initialize real-time visualizer if enabled
    visualizer: Optional[RealtimeVisualizer] = None
    if viz_enabled:
        visualizer = RealtimeVisualizer(
            title=f"Training Drifting Model",
            target_fps=viz_target_fps,
            max_display_points=viz_max_points,
            loss_history_size=viz_history_size,
        )
        print(f"Real-time visualization enabled (target: {viz_target_fps} FPS)")
    else:
        print("Real-time visualization disabled")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_iter)

    for i in pbar:
        pos = data_fn(batch_size)

        loss = model(pos, n_gen=batch_size).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            pbar.set_description(f"loss: {loss.item():.4e}")
        
        # Update real-time visualizer
        if visualizer is not None and (i + 1) % viz_update_every == 0:
            with torch.no_grad():
                gen_vis = model.generate(min(500, viz_max_points))
                real_vis = data_fn(min(500, viz_max_points))
                visualizer.update(i + 1, loss.item(), real_vis, gen_vis)
            
            # Check if window was closed
            if visualizer.is_closed():
                print("\nVisualization window closed. Stopping training.")
                break

        # Static matplotlib exports
        if (i + 1) % sample_every == 0:
            model.eval()

            # Sample comparison
            gen_vis = model.generate(n_samples)
            real_vis = data_fn(n_samples)
            viz_comparison(real_vis, gen_vis, i + 1)

            # Drift field visualization
            gen_drift = model.generate(200)
            pos_drift = data_fn(2000)
            V = compute_drift(gen_drift, pos_drift, gen_drift, temp=model.temp)
            viz_drift_field(gen_drift, pos_drift, V)

            model.train()
    
    # Close visualizer
    if visualizer is not None:
        visualizer.close()
