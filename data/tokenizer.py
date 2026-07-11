from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = list(corpus)
        merges = []

        for _ in range(num_merges):
            pairs = {}

            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                pairs[pair] = pairs.get(pair,0) + 1

            if not pairs:
                break 

            most_frequent = max(pairs.values())

            candidates = sorted(pair for pair, count in pairs.items() if count == most_frequent)

            best = candidates[0]

            merges.append([best[0],best[1]])

            tokens = []
            i = 0

            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == best[0] and chars[i+1] == best[1]:
                    tokens.append(best[0] + best[1])
                    i+=2
                else:
                    tokens.append(chars[i])
                    i+=1
            
            chars = tokens

        return merges




        
