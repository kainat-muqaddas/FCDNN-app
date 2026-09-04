"""
POD-FCDNN Surrogate Model Engine
Core execution logic for model loading, inference, and snapshot reconstruction.
"""

import os
import torch
import torch.nn as nn
import numpy as np


class POD_FCDNN(nn.Module):
    """Deep Neural Network architecture for mapping parameters to POD coefficients."""
    def __init__(self, input_dim=1, output_dim=10, hidden_units=[64, 128, 128, 64]):
        super(POD_FCDNN, self).__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_units:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SurrogateTrainer:
    """Trainer container for model state and POD bases."""
    def __init__(self, model, spatial_modes, mean_vector, xy_coords):
        self.model = model
        self.spatial_modes = spatial_modes  # Shape: (N_points * N_vars, N_modes)
        self.mean_vector = mean_vector      # Shape: (N_points * N_vars,)
        self.xy_coords = xy_coords          # Shape: (N_points, 2)
        self.model.eval()

    def predict(self, param_val):
        """Predict POD coefficients for a given parameter input."""
        with torch.no_grad():
            inp = torch.tensor([[float(param_val)]], dtype=torch.float32)
            coeffs = self.model(inp).numpy().squeeze()
        return coeffs

    def reconstruct(self, coeffs):
        """Reconstruct full flow field snapshot from predicted POD coefficients."""
        # Field = Mean + Sum(coeff_i * mode_i)
        reconstruction = self.mean_vector + np.dot(self.spatial_modes, coeffs)
        
        n_points = len(self.xy_coords)
        u = reconstruction[0 : n_points]
        v = reconstruction[n_points : 2 * n_points]
        p = reconstruction[2 * n_points : 3 * n_points]

        return {
            "u": u,
            "v": v,
            "p": p,
            "xy": self.xy_coords,
        }


def load_checkpoint(checkpoint_path):
    """Load checkpoint dictionary and initialize trainer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    spatial_modes = checkpoint["spatial_modes"]
    mean_vector = checkpoint["mean_vector"]
    xy_coords = checkpoint["xy_coords"]
    
    n_modes = spatial_modes.shape[1]
    
    model = POD_FCDNN(input_dim=1, output_dim=n_modes)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    trainer = SurrogateTrainer(
        model=model,
        spatial_modes=spatial_modes,
        mean_vector=mean_vector,
        xy_coords=xy_coords,
    )
    
    return trainer


def predict_and_reconstruct(trainer, param_val):
    """Helper function to perform end-to-end inference."""
    coeffs = trainer.predict(param_val)
    return trainer.reconstruct(coeffs)
