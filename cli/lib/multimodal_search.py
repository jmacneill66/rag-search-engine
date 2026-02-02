# multimodal_search.py
"""
Multimodal search utilities using CLIP via sentence-transformers.
Supports both text → text and image → text (movie) search.
"""
from pathlib import Path
from typing import List, Dict, Union, Any
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from lib.search_utils import cosine_similarity_batch
import numpy as np


class MultimodalSearch:
    """
    Wrapper for CLIP-based multimodal embeddings with support for movie search.
    """
    def __init__(self, documents: List[Dict[str, Any]], model_name: str = "clip-ViT-B-32"):
        """
        Initialize with movie documents and load CLIP model.
        Args:
            documents: List of movie dicts, each with at least 'id', 'title', 'description'
            model_name: CLIP model from sentence-transformers
        """
        print(f"Loading CLIP model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")
        self.documents = documents
        # Prepare concatenated text for each movie: "Title: Description"
        self.texts: List[str] = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        # Pre-compute text embeddings once (much faster for repeated searches)
        print("Encoding movie descriptions...")
        self.text_embeddings = self.model.encode(
            self.texts,
            convert_to_tensor=True,          # returns torch.Tensor → better for util.cos_sim
            show_progress_bar=True
        )
        print(f"Encoded {len(self.texts)} movies.")

    def embed_image(self, image_path: Union[str, Path]) -> List[float]:
        """
        Generate CLIP embedding for a single image (used in previous tasks).
        """
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        embedding = self.model.encode([img], show_progress_bar=False)[0]
        return embedding.tolist()

    def search_with_image(self, image_path: Union[str, Path], top_k: int = 5) -> List[Dict]:
        """
        Find top-k movies most similar to the given image using CLIP embeddings.
        """
        # Load and encode image (returns numpy array by default)
        img = Image.open(Path(image_path).expanduser().resolve()).convert("RGB")
        # We want numpy for your cosine function
        image_embedding = self.model.encode([img], show_progress_bar=False)[0]          # shape (dim,)
        # → either store them as numpy in __init__, or convert here
        text_emb_np = self.text_embeddings.cpu().numpy() if hasattr(self.text_embeddings, 'cpu') else self.text_embeddings
        # Compute similarities (now shape (n_movies,))
        similarities = cosine_similarity_batch(image_embedding, text_emb_np)
        # Convert to torch just for convenient .topk() (or use np.argsort instead)
        import torch
        similarities_t = torch.from_numpy(similarities)
        top_results = similarities_t.topk(top_k)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            doc = self.documents[int(idx)]
            results.append({
                "id": doc.get("id", "unknown"),
                "title": doc["title"],
                "description": doc["description"],
                "similarity": round(float(score), 3)
            })
        return results

def verify_image_embedding(image_path: Union[str, Path]) -> None:
    embedder = MultimodalSearch(documents=[])  
    embedding = embedder.embed_image(image_path)
    print(f"Embedding shape: {len(embedding)} dimensions")

def image_search_command(image_path: Union[str, Path]) -> List[Dict]:
    """
    Top-level function for CLI: load movie data → search with image → return results.
    """
    # Load movie dataset 
    from lib.search_utils import load_movies  
    movies = load_movies()  # should return list of dicts with 'id', 'title', 'description'
    if not movies:
        raise ValueError("No movies loaded – check your data source")
    search_engine = MultimodalSearch(documents=movies)
    results = search_engine.search_with_image(image_path)
    return results