import json
import os
from pathlib import Path
import numpy as np

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5  # k1 is a tunable saturation parameter that controls the diminishing returns (a common value is 1.5)
BM25_B = 0.75  # b is a tunable normalization parameter that controls how much we care about document length 
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1
SCORE_PRECISION = 4

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"  
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
MOVIE_EMBEDDINGS_PATH = PROJECT_ROOT / "cache" / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = PROJECT_ROOT / "cache" / "chunk_metadata.json"

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def euclidean_norm(vec):
    total = 0.0
    for x in vec:
        total += x**2
    return total**0.5

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
    
def cosine_similarity_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between ONE query vector and MANY document vectors.
        Args:
        query:  shape (dim,)
        corpus: shape (n, dim)
        Returns:
        similarities: shape (n,) — one score per document
    """
    # Normalize query once
    norm_query = np.linalg.norm(query)
    if norm_query == 0:
        return np.zeros(corpus.shape[0], dtype=np.float32)

    # Normalize all corpus vectors (axis=1 = per row)
    norms_corpus = np.linalg.norm(corpus, axis=1)
    # Avoid division by zero (very rare with CLIP but good practice)
    norms_corpus[norms_corpus == 0] = 1e-12

    # Dot product: corpus @ query → shape (n,)
    dots = np.dot(corpus, query)

    return dots / (norms_corpus * norm_query)



