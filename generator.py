from typing import Dict, Optional, Set, Tuple

import numpy as np
from tqdm.notebook import tqdm


class WordGenerator:
    def __init__(
            self,
            alphabet: Set[str],
            lengths: Tuple[int, int] = (8, 16),
            positives: float = 0.15,
            sharpness: float = 1.0,
            seed: Optional[int] = None
    ):
        self.alphabet = sorted(alphabet)
        self.tokens = len(self.alphabet)
        self.sharpness = sharpness
        self.lengths = lengths

        np.random.seed(seed)
        self.initial_probability = self.create_probability_vector()
        self.matrix = self.create_transition_matrix()

        self.positives = positives
        self.word_probability = self.create_probability_vector()

    def __call__(self, n: int) -> Dict[str, bool]:
        words_with_weights = self.get_words_with_weights(n)
        weights = np.array(list(words_with_weights.values()))
        weights /= weights.sum()
        positive_items = round(len(words_with_weights) * self.positives)
        positives = set(np.random.choice(
            list(words_with_weights.keys()), positive_items,
            replace=False, p=weights
        ))

        return {
            word: word in positives
            for word in words_with_weights
        }

    def get_words_with_weights(self, n: int) -> Dict[str, float]:
        words = {}
        for _ in tqdm(range(n)):
            word_length = np.random.randint(*self.lengths)
            token = np.random.choice(self.tokens, p=self.initial_probability)
            prob = self.word_probability[token]
            word = self.alphabet[token]
            for i in range(max(1, word_length - 1)):
                token = np.random.choice(self.tokens, p=self.matrix[token])
                word += self.alphabet[token]
                prob += self.word_probability[token]

            words[word] = prob / word_length

        return words

    def create_probability_vector(self) -> np.array:
        vector = np.random.rand(self.tokens) ** self.sharpness
        return vector / vector.sum()

    def create_transition_matrix(self) -> dict:
        matrix = np.random.rand(self.tokens, self.tokens) ** self.sharpness
        return matrix / matrix.sum(axis=1)[:, None]
