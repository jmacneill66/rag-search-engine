import os
import pickle
import math
from pathlib import Path
from .search_utils import CACHE_DIR
from collections import Counter, defaultdict
from .search_utils import load_movies
from .text_processing import tokenize_and_stem
from .search_utils import BM25_K1, BM25_B

class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = defaultdict(set)

        # doc_id -> full document
        self.docmap: dict[int, dict] = {}

        # Add these path attributes:
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

        # dict of doc IDs to Counter of term frequencies
        self.term_frequencies: dict[int, Counter] = {}

        # doc_lengths -> empty dictionary
        self.doc_lengths: dict[int, int] = {}


    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_and_stem(text)

        # Count total number of tokens in this document
        self.doc_lengths[doc_id] = len(tokens)

        # Initialize counter for this document
        tf = Counter()

        for token in tokens:
            # Inverted index
            self.index[token].add(doc_id)

            # Term frequency
            tf[token] += 1

        self.term_frequencies[doc_id] = tf

    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        docs = self.index.get(term, set())
        return sorted(docs)

    def build(self) -> None:
        movies = load_movies()

        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m

            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

            counter = self.term_frequencies[doc_id]
            total_terms = sum(counter.values())

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        with (cache_dir / "index.pkl").open("wb") as f:
            pickle.dump(self.index, f)

        with (cache_dir / "docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f)

        with (cache_dir / "term_frequencies.pkl").open("wb") as f:
            pickle.dump(self.term_frequencies, f)

        with (cache_dir / "doc_lengths.pkl").open("wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        cache_dir = Path("cache")

        with (cache_dir / "index.pkl").open("rb") as f:
            self.index = pickle.load(f)

        with (cache_dir / "docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)

        with (cache_dir / "term_frequencies.pkl").open("rb") as f:
            self.term_frequencies = pickle.load(f)

        with (cache_dir / "doc_lengths.pkl").open("rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_and_stem(term)

        if len(tokens) != 1:
            raise ValueError("Term must resolve to exactly one token")

        token = tokens[0]

        if doc_id not in self.term_frequencies:
            return 0

        return self.term_frequencies[doc_id].get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_and_stem(term)

        if len(tokens) != 1:
            raise ValueError("BM25 IDF expects a single token")

        token = tokens[0]

        # N = total number of documents
        N = len(self.docmap)

        # df = document frequency
        df = len(self.index.get(token, set()))

        # Returns bm25_tf (BM25 IDF formula)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        tokens = tokenize_and_stem(term)

        if len(tokens) != 1:
            raise ValueError("BM25 TF expects a single token")

        token = tokens[0]

        # Raw term frequency
        tf = self.get_tf(doc_id, token)
        if tf == 0:
            return 0.0

        # Document length and average document length
        dl = self.doc_lengths.get(doc_id, 0)
        avgdl = self.__get_avg_doc_length()

        if avgdl == 0:
            return 0.0

        # Length normalization
        norm = 1 - b + b * (dl / avgdl)

        # Return bm25_tf 
        return (tf * (k1 + 1)) / (tf + k1 * norm)

# Helper method calculates average doc length across all docs, and handles edge case where there are no docs
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def bm25_search(
        self,
        query: str,
        limit: int = 5,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> list[dict]:
        
        query_tokens = tokenize_and_stem(query)

        scores: dict[int, float] = {}

        for doc_id in self.docmap.keys():
            total_score = 0.0

            for token in query_tokens:
                total_score += self.bm25(doc_id, token)

            if total_score > 0:
                scores[doc_id] = total_score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:limit]:
            doc = self.docmap[doc_id]
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "score": round(score, 2),
            })

        return results

    
    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        if tf == 0:
            return 0.0

        idf = self.get_bm25_idf(term)
        return tf * idf
