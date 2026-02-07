
import argparse
import yaml
import torch
from pathlib import Path

from models.drifting_model import DriftingModel
from data.generators import gen_data, gen_checkerboard, get_device
from core.drift import compute_drift
from utils.visualizations import viz_2d_data, viz_drift_field
from train import train


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Drifting Models from Scratch")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Configuration: {config}")

    # Data selection
    data_name = config['data']['dataset']
    if data_name == "8gaussians":
        data_fn = gen_data
    elif data_name == "checkerboard":
        data_fn = gen_checkerboard
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    # Visualize data - save to outputs/data/
    print(f"Visualizing {data_name} dataset...")
    viz_2d_data(
        data_fn(8096),
        filename=f"data_{data_name}.jpg",
        category="data"
    )

    # Visualize initial drift - save to outputs/drift/
    noise_dim = config['model']['noise_dim']
    hidden_dim = config['model']['hidden_dim']
    temp = config['model']['temp']
    
    torch.manual_seed(42)
    gen_init = torch.randn(150, 2, device=device) * 2.0
    pos_init = data_fn(2000)
    
    with torch.no_grad():
        # Higher temp for better visualization
        V_init = compute_drift(gen_init, pos_init, gen_init, temp=0.2)
    
    viz_drift_field(gen_init, pos_init, V_init, filename="drift_initial.jpg")

    # Initialize model
    print("\n--- Training ---")
    model = DriftingModel(
        noise_dim=noise_dim, 
        hidden_dim=hidden_dim, 
        temp=temp
    ).to(device)

    # Train
    train_config = config['training']
    train(model, data_fn, train_config)

    # Final visualization - save to outputs/final/
    print("\n--- Generating final results ---")
    model.eval()
    gen_final = model.generate(4096)
    viz_2d_data(
        gen_final,
        filename=f"final_{data_name}.jpg",
        category="final"
    )

    # Final drift field - save to outputs/drift/
    gen_drift = model.generate(200)
    pos_drift = data_fn(2000)
    
    with torch.no_grad():
        V_final = compute_drift(gen_drift, pos_drift, gen_drift, temp=temp)
    
    viz_drift_field(
        gen_drift,
        pos_drift,
        V_final,
        filename=f"drift_final_{data_name}.jpg"
    )

    print("\nDone! Results saved to outputs/ directory.")


if __name__ == "__main__":
    main()
