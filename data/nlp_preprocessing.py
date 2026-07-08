import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive + negative
        counter = 1 
        words = sorted(set(word for sequence in combined for word in sequence.split()))

        words_with_idx = {word: idx + 1 for idx,word in enumerate(words)}

        encoded = [torch.tensor([words_with_idx[word] for word in sequence.split()]) for sequence in combined]

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)
