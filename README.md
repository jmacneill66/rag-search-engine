# RAG Search Engine

A production-grade Retrieval-Augmented Generation (RAG) system for semantic movie search, combining multiple search techniques with LLM-powered query enhancement and re-ranking.

## Strategic & Security Considerations

This project was developed with a "Security-by-Design" mindset, acknowledging the unique risks associated with RAG (Retrieval-Augmented Generation) and Agentic workflows:

Prompt Injection Mitigation: The supervisor agent architecture is designed to act as a logic gate, reducing the risk of "Direct Prompt Injection" by isolating user input from the final execution layer.

Data Privacy & Residency: The system is built to support local vector stores (FAISS/ChromaDB). For government ICT applications, this allows for sensitive documents to be processed within a controlled environment (on-prem or private cloud) rather than being sent to public LLM training sets.

Hallucination Governance: By utilizing a RAG architecture, the model is grounded in "Ground Truth" documents. This transforms the LLM from a creative generator into a retrieval-focused advisor, significantly reducing the risk of misinformation in mission-critical environments.

Secure API Management: Utilizes .env masking for all OpenAI and LangGraph API keys, ensuring that no sensitive credentials or "secrets" are exposed within the version control history.

## Features

### Search Methods

- **Semantic Search**: Dense vector embeddings using sentence transformers
- **Keyword Search (BM25)**: Traditional TF-IDF based inverted index
- **Hybrid Search**: Combines semantic and keyword search using weighted scoring or Reciprocal Rank Fusion (RRF)

### Query Enhancement Features

- **Spell Correction**: Fix typos using Gemini LLM
- **Query Rewriting**: Transform vague queries into specific, searchable terms

### Re-ranking Methods

- **Individual LLM Re-ranking**: Score each result individually (0-10)
- **Batch LLM Re-ranking**: Rank all results in a single API call
- **Cross-Encoder Re-ranking**: Fast local ML model for relevance scoring

### Advanced Features

- **Semantic Chunking**: Split long documents into meaningful chunks with overlap
- **Multimodal Query Enhancement**: Rewrite text queries using visual information from images
- **RAG Pipeline**: Comprehensive retrieval-augmented generation
- **Multi-document Summarization**: Generate 3-4 sentence overviews from search results
- **Citation Generation**: LLM answers with source citations [1], [2], etc.
- **Question Answering**: Direct, concise answers to factual questions
- **Evaluation Tools**: Precision@k and Recall@k metrics

## Installation

### Prerequisites

- Python 3.12+
- Gemini API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rag-search-engine.git
cd rag-search-engine
```

1. Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Install dependencies:

```bash
uv sync
```

1. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

1. Build initial indexes (first run will take a few minutes):

```bash
uv run cli/semantic_search_cli.py verify_embeddings
```

## Usage

### Semantic Search

```bash
# Basic semantic search
uv run cli/semantic_search_cli.py search "science fiction space adventure" --limit 10

# Chunked semantic search (better for long documents)
uv run cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 5
```

### Hybrid Search

```bash
# Weighted hybrid search (alpha controls BM25 vs semantic weight)
uv run cli/hybrid_search_cli.py weighted-search "action movie" --alpha 0.5 --limit 10

# RRF (Reciprocal Rank Fusion) search
uv run cli/hybrid_search_cli.py rrf-search "thriller mystery" -k 60 --limit 5
```

### Query Enhancement

```bash
# Spell correction
uv run cli/hybrid_search_cli.py rrf-search "brittish bear" --enhance spell --limit 5

# Query rewriting (vague → specific)
uv run cli/hybrid_search_cli.py rrf-search "that bear movie in london" --enhance rewrite --limit 5
```

### Re-ranking

```bash
# Cross-encoder re-ranking (fastest, no API calls)
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --rerank-method cross_encoder --limit 5

# Batch LLM re-ranking (single API call)
uv run cli/hybrid_search_cli.py rrf-search "scary bear movies" --rerank-method batch --limit 3

# Individual LLM re-ranking (most accurate, slowest)
uv run cli/hybrid_search_cli.py rrf-search "bear wilderness survival" --rerank-method individual --limit 3
```

### Augmented Generation

```bash
# Multi-document summarization
uv run cli/augmented_generation_cli.py summarize "movies about dinosaurs" --limit 5

# Answer with citations
uv run cli/augmented_generation_cli.py citations "action movie with lasers" --limit 5

# Direct question answering
uv run cli/augmented_generation_cli.py question "What are the best family movies about bears?" --limit 5

# Full RAG pipeline (search + generate comprehensive answer)
uv run cli/augmented_generation_cli.py rag "romantic comedies set in Europe" --top-k 5
```

### Multimodal Search (Image + Text)

```bash
# Rewrite a text query using visual information from an image
uv run cli/describe_image_cli.py \
  --image data/paddington.jpeg \
  --query "British bear movie"

# Output: "Rewritten query: Paddington bear London marmalade family film"

# Use with other search commands
REWRITTEN=$(uv run cli/describe_image_cli.py --image movie_poster.jpg --query "action movie" | grep "Rewritten query:" | cut -d: -f2)
uv run cli/hybrid_search_cli.py rrf-search "$REWRITTEN" --limit 5
```

### Evaluation

```bash
# Evaluate search quality with precision@k and recall@k
uv run cli/evaluation_cli.py --limit 10
```

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── semantic_search_cli.py      # Semantic search commands
│   ├── hybrid_search_cli.py        # Hybrid search & re-ranking
│   ├── augmented_generation_cli.py # Summarization & citations
│   ├── describe_image_cli.py       # Multimodal image query rewriting
│   └── evaluation_cli.py           # Search quality metrics
├── cli/lib/
│   ├── semantic_search.py          # Semantic search implementation
│   ├── keyword_search.py           # BM25 inverted index
│   ├── hybrid_search.py            # Hybrid search & re-ranking
│   ├── text_processing.py         # Tokenization & stemming
│   └── search_utils.py             # Shared utilities
├── data/
│   ├── movies.json                 # Movie dataset
│   └── golden_dataset.json         # Evaluation test cases
├── cache/                          # Generated indexes & embeddings
└── README.md
```

## Technical Details

### Search Pipeline

1. **Indexing Phase** (one-time setup):
   - Build inverted index for BM25 search
   - Generate semantic embeddings for documents
   - Chunk long documents and embed chunks

2. **Query Phase**:
   - Optional: Enhance query (spell check, rewrite)
   - Run parallel searches (BM25 + semantic)
   - Combine results using RRF or weighted scoring
   - Optional: Re-rank using cross-encoder or LLM

3. **Generation Phase** (optional):
   - Retrieve top-k documents
   - Generate summary or answer with citations using LLM

### Models Used

- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions)
- **Cross-Encoder**: `cross-encoder/ms-marco-TinyBERT-L2-v2`
- **LLM**: Gemini 2.5 Flash (via Google AI API)

### Performance

- **Dataset**: ~4,800 movies
- **Chunk Embeddings**: ~72,900 chunks
- **Search Latency**: <100ms (without re-ranking)
- **With Cross-Encoder**: ~200-300ms
- **With LLM Re-ranking**: 3-15 seconds (batch) or 30-60 seconds (individual)

## Examples

### Example 1: Combined Enhancement + Re-ranking

```bash
uv run cli/hybrid_search_cli.py rrf-search \
  "famly movee abut bares in the woods" \
  --enhance spell \
  --rerank-method cross_encoder \
  --limit 5
```

Output:

```
Enhanced query (spell): 'famly movee abut bares in the woods' -> 'family movie about bears in the woods'

Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):

1. The Berenstain Bears' Christmas Tree
   Cross Encoder Score: 4.521
   RRF Score: 0.027
   BM25 Rank: 37, Semantic Rank: 1
   ...
```

### Example 2: RAG Pipeline

```bash
uv run cli/augmented_generation_cli.py rag "scary movies with bears" --top-k 5
```

Output:

```
Searching for: 'scary movies with bears' ...

======================================================================
Search Results:
  - Into the Grizzly Maze
  - The Edge
  - Backcountry
  - Grizzly
  - Unnatural

RAG Response:
For Hoopla users seeking thrilling bear-themed horror, several options stand out. 
"Into the Grizzly Maze" features two estranged brothers confronting a rogue grizzly 
in the Alaskan wilderness. "The Edge" combines survival drama with horror as Anthony 
Hopkins and Alec Baldwin face off against a massive Kodiak bear. "Backcountry" offers 
a realistic, terrifying portrayal of a couple's camping trip gone wrong when they 
encounter a black bear. These films deliver intense wilderness survival scenarios 
with bears as the primary antagonists.
======================================================================
```

### Example 3: Multimodal Query Enhancement

```bash
# Use an image to improve search query
uv run cli/describe_image_cli.py \
  --image data/movie_poster.jpg \
  --query "bear movie"
```

Output:

```
Rewritten query: Paddington bear London marmalade sandwich family comedy film
Total tokens: 45
```

This enhanced query can then be used for more accurate search results.

## Evaluation Results

On the golden dataset (6 test queries):

- **Precision@5**: 0.45 average
- **Recall@5**: 0.78 average

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built as part of the Boot.dev RAG course
- Movie dataset from Hoopla streaming service
- Embeddings from SentenceTransformers library
- LLM capabilities via Google Gemini API

## Contact

Project Link: [https://github.com/jmacneill66/rag-search-engine](https://github.com/jmacneill66/rag-search-engine)
