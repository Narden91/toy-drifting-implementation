
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
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Drifting Models from Scratch")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
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

    # Visualize data
    print(f"Visualizing {data_name} dataset...")
    viz_2d_data(data_fn(8096), filename=f"data_{data_name}.jpg")

    # Visualize initial drift
    # Create temp model just to check dimensions/temp or just use params
    noise_dim = config['model']['noise_dim']
    hidden_dim = config['model']['hidden_dim']
    temp = config['model']['temp']
    
    torch.manual_seed(42)
    gen_init = torch.randn(150, 2, device=device) * 2.0
    pos_init = data_fn(2000)
    with torch.no_grad():
        V_init = compute_drift(gen_init, pos_init, gen_init, temp=0.2) # Higher temp for viz
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

    # Final visualization
    model.eval()
    gen_final = model.generate(4096)
    viz_2d_data(gen_final, filename=f"final_{data_name}.jpg")

    gen_drift = model.generate(200)
    pos_drift = data_fn(2000)
    with torch.no_grad():
        V_final = compute_drift(gen_drift, pos_drift, gen_drift, temp=temp)
    viz_drift_field(gen_drift, pos_drift, V_final, filename=f"drift_final_{data_name}.jpg")

    print("Done.")

if __name__ == "__main__":
    main()