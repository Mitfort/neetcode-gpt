from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        
        stoi = {}
        text = sorted(set(text))
        for i, letter in enumerate(text):
            stoi[letter] = i
        
        itos = {}

        for key,val in stoi.items():
            itos[val] = key

        return (stoi, itos)


    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded = []
        for letter in text:
            encoded.append(stoi[letter])

        return encoded

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded = ""

        for idx in ids: 
            decoded += itos[idx]

        return decoded
