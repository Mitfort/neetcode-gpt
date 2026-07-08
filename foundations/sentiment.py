import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.emb = nn.Embedding(vocabulary_size,16)
        self.linear = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embeded = self.emb(x)
        embeded_mean = embeded.mean(dim=1)
        # Return a B, 1 tensor and round to 4 decimal places
        logits = self.linear(embeded_mean)
        yhat = self.sigmoid(logits)

        return torch.round(yhat, decimals=4)
        
