import re
from collections import Counter
from typing import List
import numpy as np

class BagOfWords:
    def __init__(self):
        self.vocabulary_ = {}

    def _tokenize(self, text: str) -> List[str]:
        # Use a more general word character regex to match test expectations.
        # This regex should split "C3PO's" into "c3po" and "s"
        # and include accented characters and numbers.
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents: List[str]):
        all_tokens = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.update(tokens)

        sorted_tokens = sorted(list(all_tokens))
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}
        return self

    def transform(self, document: str) -> np.ndarray:
        if not hasattr(self, 'vocabulary_') or not self.vocabulary_:
            raise AttributeError("BagOfWords model has not been fitted. Call 'fit' before 'transform'.")

        doc_tokens = self._tokenize(document)
        token_counts = Counter(doc_tokens)

        bow_vector = np.zeros(len(self.vocabulary_), dtype=int)

        for token, count in token_counts.items():
            if token in self.vocabulary_:
                idx = self.vocabulary_[token]
                bow_vector[idx] = count
        return bow_vector