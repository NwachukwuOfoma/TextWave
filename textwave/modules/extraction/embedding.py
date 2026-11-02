# Use this code for: {ROOT_DIR}/textwave/modules/retrieval/embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np

class Embedding:
    """
    A class to handle the embedding of text documents and queries using a pre-trained model.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the Embedding class and loads the sentence transformer model.
        :param model_name: The name of the sentence-transformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def get_embedding_dim(self) -> int:
        """
        Returns the embedding dimension of the loaded model. This is crucial for FAISS.
        """
        # Get the embedding dimension from the model configuration
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """
        Embeds a list of documents into vectors.
        :param documents: A list of strings to embed.
        :return: A numpy array of embeddings.
        """
        # The `convert_to_numpy=True` is efficient for FAISS
        return self.model.encode(documents, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into a vector.
        :param query: The query string to embed.
        :return: The numpy array embedding for the query.
        """
        return self.model.encode(query, convert_to_numpy=True)

if __name__ == "__main__":
    embedding_model_name = 'all-MiniLM-L6-v2'
    embedding_model = Embedding(embedding_model_name)

    # Example for embedding a single query
    query = "Who suggested Lincoln grow a beard?"
    query_embedding = embedding_model.embed_query(query)
    print("Query Embedding Shape:", query_embedding.shape)

    # Example for embedding multiple documents
    docs = [
        "Grace Bedell, an 11-year-old girl, wrote to Lincoln in 1860.",
        "She suggested that growing a beard would improve his appearance.",
        "Lincoln responded to her letter and later met her."
    ]
    doc_embeddings = embedding_model.embed_documents(docs)
    print("Document Embeddings Shape:", doc_embeddings.shape)

    print("Embedding dimension:", embedding_model.get_embedding_dim())