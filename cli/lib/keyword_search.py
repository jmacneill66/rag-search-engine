from .inverted_index import InvertedIndex
from .text_processing import tokenize_and_stem


def search_command(query: str, limit: int = 5) -> list[dict]:
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    query_tokens = tokenize_and_stem(query)

    results = []
    seen_ids = set()

    for token in query_tokens:
        doc_ids = index.get_documents(token)

        for doc_id in doc_ids:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(index.docmap[doc_id])

            if len(results) >= limit:
                return results

    return results

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)