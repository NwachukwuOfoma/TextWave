# Create this new file at: {ROOT_DIR}/textwave/modules/retrieval/indexing.py

import faiss
import numpy as np

class FaissIndexer:
    """
    A class for creating and managing a FAISS index for vector similarity search.
    """
    def __init__(self, dim: int, strategy: str = "bruteforce"):
        """
        Initializes the FaissIndexer.
        :param dim: The dimensionality of the vectors to be indexed.
        :param strategy: The FAISS index strategy to use. Currently supports 'bruteforce'.
        """
        self.dim = dim
        self.strategy = strategy
        self.index = None

    def create_index(self) -> faiss.Index:
        """
        Creates a FAISS index based on the specified strategy.
        :return: The created FAISS index.
        """
        if self.strategy == "bruteforce":
            # IndexFlatL2 is a brute-force index that computes exact L2 distances.
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError(f"Unknown FAISS strategy: {self.strategy}")
        return self.index

    def add_to_index(self, vectors: np.ndarray):
        """
        Adds vectors to the FAISS index.
        :param vectors: A numpy array of vectors to add.
        """
        if self.index is None:
            raise RuntimeError("Index has not been created. Call create_index() first.")
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors).astype('float32')
        if vectors.dtype != 'float32':
             vectors = vectors.astype('float32')

        self.index.add(vectors)