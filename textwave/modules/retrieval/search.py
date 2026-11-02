import faiss
import numpy as np

class FaissSearcher:
    """A class to perform similarity searches on a FAISS index."""
    def __init__(self, index: faiss.Index):
        if not index.is_trained:
            raise ValueError("The provided FAISS index is not trained.")
        self.index = index

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for the top-k most similar vectors.
        :param query_vectors: A numpy array of query vectors.
        :param k: The number of nearest neighbors to retrieve.
        :return: A tuple containing distances and indices of the top-k neighbors.
        """
        if query_vectors.dtype != 'float32':
            query_vectors = query_vectors.astype('float32')
        
        # Ensure k is not greater than the number of items in the index
        if k > self.index.ntotal:
            k = self.index.ntotal

        # FIX: Return both distances and indices, as is standard.
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices