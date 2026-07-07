import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        results = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    mean = round(torch.mean(x).item(),4)
                    std = round(torch.std(x).item(),4)

                    dead = torch.sum(torch.all(x <= 0, dim=0))
                    dead_fraction = round((dead / x.shape[1]).item(),4)

                    results.append({"mean": mean, "std": std, "dead_fraction": dead_fraction})

        return results            

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        
        results = []

        yhat = model(x)

        loss = nn.MSELoss()(yhat,y)
        loss.backward()

        for layer in model.children():
            
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                mean = round(torch.mean(grad).item(),4)
                std = round(torch.std(grad).item(),4)

                norm = round(torch.norm(grad).item(),4)

                results.append({
                    "mean": mean,
                    "std": std,
                    "norm": norm
                })

        return results

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        for stats in activation_stats:
            if stats["dead_fraction"] > 0.5:
                return "dead_neurons"

        for stats in gradient_stats:
            if stats["norm"] > 1000: 
                return "exploding_gradients"
        
        if gradient_stats[-1]['norm'] < 1e-5:
            return "vanishing_gradients"

        for stats in activation_stats:
            if stats['std'] < 0.1:
                return 'vanishing_gradients'
            elif stats['std'] > 10.0:
                return 'exploding_gradients'

        return "healthy"
