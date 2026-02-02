import numpy as np
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    SCORE_PRECISION,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        """Generate a semantic embedding for the given text."""
        if not text or text.strip() == "":
            raise ValueError("Text cannot be empty or contain only whitespace")
        # Encode the text (wrap in list, get first result)
        embedding = self.model.encode([text])[0]
        return embedding
    
    def build_embeddings(self, documents):
        """Build embeddings for a list of documents."""
        self.documents = documents  
        # Build document_map with id as key
        for doc in documents:
            self.document_map[doc['id']] = doc
        # Create string representations of each movie
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        # Generate embeddings with progress bar
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        # Save embeddings to disk
        Path('cache').mkdir(exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        """Load embeddings from cache or create them if they don't exist."""
        self.documents = documents
        # Build document_map
        for doc in documents:
            self.document_map[doc['id']] = doc
        # Check if cached embeddings exist
        if MOVIE_EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            # Verify the embeddings match the documents
            if len(self.embeddings) == len(documents):
                return self.embeddings
        # Otherwise, rebuild from scratch
        return self.build_embeddings(documents)

    def search(self, query, limit):
        """Search for documents similar to the query."""
        # 1. Check if embeddings are loaded
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")        
        # 2. Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        # 3. Calculate cosine similarity between query and each document
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        # 4. Create list of (similarity_score, document) tuples
        results = []
        for i, similarity in enumerate(similarities):
            results.append((similarity, self.documents[i]))
        # 5. Sort by similarity score in descending order
        results.sort(key=lambda x: x[0], reverse=True)
        # 6. Return top results (up to limit)
        top_results = []
        for score, doc in results[:limit]:
            top_results.append({
                'score': score,
                'title': doc['title'],
                'description': doc['description']
            })
        return top_results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        """Build embeddings for document chunks."""
        # Populate documents and document_map
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc     
        all_chunks = []
        chunk_metadata = []
        # Process each document
        for movie_idx, doc in enumerate(documents):
            description = doc.get('description', '')
            # Skip empty descriptions
            if not description.strip():
                continue
            # Split into semantic chunks using shared defaults
            chunks = semantic_chunk(
                description,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP
            )
            # Add each chunk and its metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'movie_idx': movie_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks)
                })
        # Generate embeddings for all chunks
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        # Save to cache
        Path('cache').mkdir(exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(all_chunks)
            }, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        """Load chunk embeddings from cache or create them."""
        # Populate documents and document_map
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        # Check if cache exists
        embeddings_path = Path(CHUNK_EMBEDDINGS_PATH)
        metadata_path = Path(CHUNK_METADATA_PATH)
        if embeddings_path.exists() and metadata_path.exists():
            # Load from cache
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as f:
                data = json.load(f)
                self.chunk_metadata = data['chunks']
            return self.chunk_embeddings
        # Otherwise, build from scratch
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query, limit=10):
        """Search for document chunks similar to the query and aggregate by movie."""
        # Check if chunk embeddings are loaded
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")    
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        # Populate empty list to store chunk score dictionaries
        chunk_scores = []
        # For each chunk embedding, calculate similarity
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)     
            # Get metadata for this chunk
            metadata = self.chunk_metadata[i]
            # Append chunk score dictionary
            chunk_scores.append({
                'chunk_idx': metadata['chunk_idx'],
                'movie_idx': metadata['movie_idx'],
                'score': similarity
            })
        # Create dictionary mapping movie_idx to best score
        movie_scores = {}
        # For each chunk score, keep the highest score per movie
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            score = chunk_score['score']
            # Update if this is the first score or if it's higher
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]['score']:
                movie_scores[movie_idx] = chunk_score
        # Sort movie scores by score in descending order
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        # Filter to top limit movies
        top_movies = sorted_movies[:limit]
        # Format results
        results = []
        SCORE_PRECISION = 4  # Number of decimal places for score
        for movie_idx, chunk_info in top_movies:
            doc = self.documents[movie_idx]    
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'document': doc['description'][:100],
                'score': round(chunk_info['score'], SCORE_PRECISION),
                'metadata': {
                    'chunk_idx': chunk_info['chunk_idx'],
                    'movie_idx': movie_idx
                }
            })
        return results
    

def verify_model():
    """Verify the model is loaded correctly and print its information."""
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    """Generate and return the embedding for the given text."""
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    """Verify embeddings are loaded/created correctly."""
    search = SemanticSearch()
    # Load documents from movies.json
    with open('data/movies.json', 'r') as f:
        data = json.load(f)
    # Extract the movies list
    documents = data['movies']
    # Load or create embeddings
    embeddings = search.load_or_create_embeddings(documents)
    # Print verification info
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    """Generate and return the embedding for the given query text."""
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding

def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    return [a + b for a, b in zip(vec1, vec2)]

def subtract_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    return [a - b for a, b in zip(vec1, vec2)]  

def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("vectors must be the same length")
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total

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

def semantic_chunk_text(text, max_chunk_size=4, overlap=0):
    """CLI helper: chunk text and return chunks for printing."""
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    return chunks

def semantic_chunk(text, max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """Split text into semantic chunks by sentences with improved edge case handling."""
    # Strip leading and trailing whitespace from input
    text = text.strip()
    # If nothing left after stripping, return empty list
    if not text:
        return []
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Handle single sentence without punctuation
    if len(sentences) == 1 and not re.search(r'[.!?]$', sentences[0].strip()):
        # Treat the whole text as one sentence
        return [text]
    # Strip whitespace from each sentence
    sentences = [s.strip() for s in sentences if s.strip()]
    # If no sentences remain, return empty list
    if not sentences:
        return []
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        # Skip if this chunk is entirely within the overlap of the previous chunk
        if chunks and len(chunk_sentences) <= overlap:
            break
        # Join sentences and strip whitespace
        chunk = " ".join(chunk_sentences).strip()
        # Only add chunks that have content after stripping
        if chunk:
            chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks