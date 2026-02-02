import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from dotenv import load_dotenv
import logging
logging.basicConfig(
    level=logging.DEBUG,              # or INFO if you want less noise
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def normalize(self, scores):
        """Normalize scores using min-max normalization."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [1.0] * len(scores)
        normalized = []
        for score in scores:
            normalized.append((score - min_score) / (max_score - min_score))
        return normalized

    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha=0.5, limit=5):
        """Perform weighted hybrid search combining BM25 and semantic search."""
        # Get many more results than we need (500x)
        fetch_k = limit * 500
        # 1. Get BM25 results and normalize
        bm25_results = self._bm25_search(query, fetch_k)
        bm25_results = normalize_search_results(bm25_results)
        # 2. Get semantic results and normalize
        semantic_results = self.semantic_search.search_chunks(query, fetch_k)
        semantic_results = normalize_search_results(semantic_results)
        # 3. Create dictionary mapping doc_id to combined info
        doc_info = {}
        # Add BM25 results
        for result in bm25_results:
            doc_id = result["id"]
            doc_info[doc_id] = {
                "title": result["title"],
                "description": self.documents[doc_id - 1]["description"],  # Get full description
                "bm25_norm": result["normalized_score"],
                "semantic_norm": 0.0
            }
        # Add/update with semantic results
        for result in semantic_results:
            doc_id = result["id"]
            if doc_id not in doc_info:
                # Get the document from self.documents for title and description
                doc = self.documents[doc_id - 1]  # Assuming 1-based IDs
                doc_info[doc_id] = {
                    "title": result["title"],
                    "description": doc["description"],
                    "bm25_norm": 0.0,
                    "semantic_norm": result["normalized_score"]
                }
            else:
                doc_info[doc_id]["semantic_norm"] = result["normalized_score"]
        # 4. Calculate hybrid scores
        scored_docs = []
        for doc_id, info in doc_info.items():
            hybrid = self.hybrid_score(
                info["bm25_norm"],
                info["semantic_norm"],
                alpha
            )
            scored_docs.append({
                "doc_id": doc_id,
                "title": info["title"],
                "description": info["description"],
                "hybrid_score": hybrid,
                "bm25_norm": info["bm25_norm"],
                "semantic_norm": info["semantic_norm"]
            })
        # 5. Sort by hybrid score descending
        scored_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
        # 6. Return top limit results
        return scored_docs[:limit]

    def rrf_score(self, ranks):
        """
        Calculate RRF score for a document given its ranks from different retrievers.
        ranks: list of rank values (1-based), or empty if not found in that retriever
        """
        k = 60  # standard constant – can be made configurable later if needed
        score = 0.0
        for rank in ranks:
            if rank is not None:  # only add if document appeared in that list
                score += 1.0 / (k + rank)
        return score

    def rrf_search(self, query, k=60, limit=5):
        """
        Reciprocal Rank Fusion hybrid search with debug logging at each stage.
        """
        logger.debug("=== Starting RRF Search ===")
        # logger.debug(f"Parameters: query='{query}', k={k}, limit={limit}")
        # print(f"→ Original query: {query!r}")

        # ────────────────────────────────────────────────
        # Stage 1: Original query
        # ────────────────────────────────────────────────
        logger.info(f"Original query: {query!r}")
        # print(f"Original query     : {query!r}")

        # ────────────────────────────────────────────────
        # Stage 2: Query after enhancements (if any)
        # ────────────────────────────────────────────────
        # If you have query rewriting / expansion / hyde / etc in the future,
        # put the enhanced version here. For now we assume it's the same.
        enhanced_query = query   # ← replace with actual enhancement when you add it

        logger.info(f"Enhanced query : {enhanced_query!r}")
        # print(f"Enhanced query     : {enhanced_query!r}")

        fetch_k = limit * 500

        # ────────────────────────────────────────────────
        # Stage 3: Raw retriever results
        # ────────────────────────────────────────────────
        bm25_raw = self._bm25_search(query, fetch_k)
        semantic_raw = self.semantic_search.search_chunks(query, fetch_k)

        logger.debug(f"BM25 retrieved {len(bm25_raw)} documents (asked for {fetch_k})")
        logger.debug(f"Semantic retrieved {len(semantic_raw)} documents (asked for {fetch_k})")
        # print(f"BM25 raw count     : {len(bm25_raw)}")
        # print(f"Semantic raw count : {len(semantic_raw)}")

        # Optional: log top few ids/titles for debugging
        if bm25_raw:
            logger.debug("BM25 top 3 ids: " + ", ".join(str(r["id"]) for r in bm25_raw[:3]))
        if semantic_raw:
            logger.debug("Semantic top 3 ids: " + ", ".join(str(r["id"]) for r in semantic_raw[:3]))

        # ────────────────────────────────────────────────
        # Stage 4: Build doc_data + ranks
        # ────────────────────────────────────────────────
        doc_data = {}

        for rank, res in enumerate(bm25_raw, 1):
            doc_id = res["id"]
            if doc_id not in doc_data:
                full_doc = self.documents[doc_id - 1]
                doc_data[doc_id] = {
                    "title": res["title"],
                    "description": full_doc["description"],
                    "bm25_rank": None,
                    "semantic_rank": None
                }
            doc_data[doc_id]["bm25_rank"] = rank

        for rank, res in enumerate(semantic_raw, 1):
            doc_id = res["id"]
            if doc_id not in doc_data:
                full_doc = self.documents[doc_id - 1]
                doc_data[doc_id] = {
                    "title": res["title"],
                    "description": full_doc["description"],
                    "bm25_rank": None,
                    "semantic_rank": None
                }
            doc_data[doc_id]["semantic_rank"] = rank

        logger.info(f"Unique documents after union: {len(doc_data)}")
        # print(f"Union doc count    : {len(doc_data)}")

        # ────────────────────────────────────────────────
        # Stage 5: After RRF scoring (before sorting)
        # ────────────────────────────────────────────────
        scored = []
        for doc_id, info in doc_data.items():
            ranks = [info["bm25_rank"], info["semantic_rank"]]
            rrf = self.rrf_score([r for r in ranks if r is not None])
            scored.append({
                "doc_id": doc_id,
                "title": info["title"],
                "description": info["description"],
                "rrf_score": rrf,
                "bm25_rank": info["bm25_rank"],
                "semantic_rank": info["semantic_rank"]
            })

        logger.debug(f"Computed RRF scores for {len(scored)} documents")

        # ────────────────────────────────────────────────
        # Stage 6: After sorting (final candidates before slicing)
        # ────────────────────────────────────────────────
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Nice debug view of top candidates with ranks & score
        if scored:
            logger.info("Top RRF candidates (before final limit):")
            for i, item in enumerate(scored[:min(8, len(scored))], 1):
                logger.info(
                    f"  {i:2d}. doc={item['doc_id']:4d}  "
                    f"score={item['rrf_score']:.5f}  "
                    f"bm25={item['bm25_rank'] or '-':>3}  "
                    f"semi={item['semantic_rank'] or '-':>3}  "
                    f"title={item['title'][:60]!r}"
                )
        else:
            logger.warning("No documents scored – check retrievers")

        # print("\nFinal RRF ranking preview:")
        # for i, item in enumerate(scored[:limit], 1):
        #     print(f"  {i}. {item['doc_id']} | {item['rrf_score']:.4f} | {item['title']}")

        # ────────────────────────────────────────────────
        # Stage 7: Final results after slicing
        # ────────────────────────────────────────────────
        final_results = scored[:limit]

        logger.info(f"Returning top {len(final_results)} / {limit} results")
        # print(f"→ Final result count: {len(final_results)}")

        logger.debug("=== Finished RRF Search ===")

        return final_results

def enhance_query_with_spell_correction(query: str) -> str:
    """Use Gemini to fix spelling errors in the query."""
    from google import genai
    from dotenv import load_dotenv
    import os
    # Load environment variables from .env file
    load_dotenv()
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    # Prompt for spell correction
    prompt = f"""Fix any spelling errors in this movie search query.
    Only correct obvious typos. Don't change correctly spelled words.
    Query: "{query}"
    If no errors, return the original query.
    Corrected:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    corrected_query = response.text.strip()
    # Remove any quotes that Gemini might add
    corrected_query = corrected_query.strip('"').strip("'")
    return corrected_query

def enhance_query_with_rewrite(query: str) -> str:
    """Use Gemini to rewrite vague queries to be more specific."""
    from google import genai
    from dotenv import load_dotenv
    import os 
    # Load environment variables
    load_dotenv()
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    # Prompt for query rewriting
    prompt = f"""Rewrite this movie search query to be more specific and searchable.
    Original: "{query}"
    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic
    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
    Rewritten query:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    rewritten_query = response.text.strip()
    # Remove any quotes that Gemini might add
    rewritten_query = rewritten_query.strip('"').strip("'")
    return rewritten_query

def enhance_query_with_expand(query: str) -> str:
    """Use Gemini to expand vague queries to be more specific."""
    from google import genai
    from dotenv import load_dotenv
    import os 
    # Load environment variables
    load_dotenv()
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    # Prompt for query expanding
    prompt = f"""Expand this movie search query with related terms.
    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.
    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"
    Query: "{query}"
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    expanded_query = response.text.strip()
    # Remove any quotes that Gemini might add
    expanded_query = expanded_query.strip('"').strip("'")
    return expanded_query

def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    """Re-rank results using individual LLM scoring for each document."""
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    import os
    import time
    # Load environment variables
    load_dotenv()
    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    # Configure safety settings to avoid PROHIBITED_CONTENT errors
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
    ]
    reranked_results = []
    for result in results:
        doc = result
        # Create prompt for scoring
        prompt = f"""Rate how well this movie matches the search query.
        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("description", "")[:200]}
        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness
        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.
        Score:"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=safety_settings
                )
            )
            # Extract score from response
            score_text = response.text.strip()
            # Handle cases like "10" or "10/10" or "Score: 8"
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', score_text)
            if match:
                rerank_score = float(match.group(1))
            else:
                rerank_score = 0.0
            # Add rerank score to result
            result["rerank_score"] = rerank_score
            reranked_results.append(result)
            # Sleep to avoid rate limits
            time.sleep(3)
        except Exception as e:
            print(f"Error re-ranking {doc.get('title', 'Unknown')}: {e}")
            # If scoring fails, assign a low score
            result["rerank_score"] = 0.0
            reranked_results.append(result)
    # Sort by rerank score descending
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results

def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    """Re-rank results using a single LLM call with all documents."""
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    import os
    import json
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
    ]
    # Build the document list string
    doc_list_str = ""
    for result in results:
        doc_id = result.get("doc_id")
        title = result.get("title", "")
        description = result.get("description", "")[:150]
        doc_list_str += f"ID {doc_id}: {title} - {description}\n"
    # Single LLM call with all documents
    prompt = f"""Rank these movies by relevance to the search query.
    Query: "{query}"
    Movies:
    {doc_list_str}
    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:
    [75, 12, 34, 2, 1]
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings
            )
        )
        # Parse the JSON response
        response_text = response.text.strip()
        # Clean up any markdown code blocks if present
        import re
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            response_text = match.group(0)
        ranked_ids = json.loads(response_text)
        # Create a mapping of doc_id to its rerank position (1-based)
        rerank_map = {}
        for rank, doc_id in enumerate(ranked_ids, start=1):
            rerank_map[doc_id] = rank
        # Assign rerank_rank to each result
        for result in results:
            doc_id = result.get("doc_id")
            # If the doc wasn't ranked by LLM, give it a high rank number (low priority)
            result["rerank_rank"] = rerank_map.get(doc_id, len(results) + 1)
        # Sort by rerank_rank ascending (rank 1 = best)
        results.sort(key=lambda x: x["rerank_rank"])
    except Exception as e:
        print(f"Error during batch re-ranking: {e}")
        # If batch ranking fails, return results in original order
    return results

def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    """Re-rank results using a cross-encoder model."""
    from sentence_transformers import CrossEncoder
    # Create cross-encoder instance (once per query)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # Create pairs of [query, document] for scoring
    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
    # Score all pairs at once
    scores = cross_encoder.predict(pairs)
    # Assign cross-encoder scores to each result
    for i, result in enumerate(results):
        result["cross_encoder_score"] = float(scores[i])
    # Sort by cross-encoder score descending
    results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return results

def normalize_search_results(results):
    """Add normalized_score field to search results."""
    if not results:
        return results
    scores = [r["score"] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        normalized_scores = [1.0] * len(scores)
    else:
        normalized_scores = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    # Add normalized_score to each result
    for i, result in enumerate(results):
        result["normalized_score"] = normalized_scores[i]
    return results