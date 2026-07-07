import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))

        weights = torch.randn(fan_out,fan_in) * std
        return [[round(v.item(), 4) for v in row] for row in weights]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
        weights = torch.randn(fan_out,fan_in) * std

        return [[round(v.item(),4) for v in row] for row in weights]


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        
        def make_weights(fan_in, fan_out):
            if init_type == "xavier":
                std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            elif init_type == "kaiming":
                std = torch.sqrt(torch.tensor(2.0 / fan_in))
            else:
                std = torch.tensor(1.0)

            return torch.randn(fan_out,fan_in) * std

        # Generate all weights first to match the expected seed sequence
        layers_weights = []
        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim
            layers_weights.append(make_weights(fan_in, fan_out))

        x = torch.randn(input_dim)
        stds = []

        for W in layers_weights:
            x = W @ x
            x = torch.relu(x)

            stds.append(round(x.std().item(), 2))

        return stds