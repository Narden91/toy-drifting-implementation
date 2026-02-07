
import torch
from tqdm import trange
from typing import Callable, Any
from torch import Tensor
from models.drifting_model import DriftingModel
from core.drift import compute_drift
from utils.visualizations import viz_comparison, viz_drift_field

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
            # Use compute_drift from core
            V = compute_drift(gen_drift, pos_drift, gen_drift, temp=model.temp)
            viz_drift_field(gen_drift, pos_drift, V)

            model.train()
