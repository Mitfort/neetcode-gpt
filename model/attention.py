import torch
import torch.nn as nn
from torchtyping import TensorType
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.K = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.Q = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.V = nn.Linear(embedding_dim,attention_dim,bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.K(embedded)
        Q = self.Q(embedded)
        V = self.V(embedded)

        attention_scores = (Q @ K.transpose(-2,-1)) / math.sqrt(K.shape[2])

        masked = torch.tril(torch.ones(attention_scores.shape))
        attention_scores = attention_scores.masked_fill(masked == 0, float("-inf"))

        scores = nn.Softmax(dim=2)(attention_scores)

        return torch.round(scores @ V, decimals=4)
