# cli/augmented_generation_cli.py
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch
from google import genai
from google.genai import types


def format_documents_for_prompt(results: list[dict]) -> str:
    """Format top search results into a clean string for the prompt"""
    lines = []
    for i, doc in enumerate(results, 1):
        title = doc.get("title", "Untitled")
        desc = doc.get("description", "").strip()
        if len(desc) > 400:
            desc = desc[:397] + "..."
        lines.append(f"{i}. {title}\n   {desc}\n")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve and pass to LLM (default: 5)")
    summarize_parser = subparsers.add_parser('summarize', help='Summarize search results')
    summarize_parser.add_argument('query', type=str, help='Search query')
    summarize_parser.add_argument('--limit', type=int, default=5, help='Number of results to retrieve (default: 5)')
    citations_parser = subparsers.add_parser('citations', help='Summarize search results and include citations')
    citations_parser.add_argument('query', type=str, help='Search query')
    citations_parser.add_argument('--limit', type=int, default=5, help='Number of results to retrieve (default: 5)')
    question_parser = subparsers.add_parser('question', help='Add a conversational question-answering command for direct responses')
    question_parser.add_argument('query', type=str, help='Search query')
    question_parser.add_argument('--limit', type=int, default=5, help='Number of results to retrieve (default: 5)')
    args = parser.parse_args()

    # Load movies (used by all commands)
    documents = load_movies()
    
    match args.command:
        case 'rag':
            query = args.query.strip()
            if not query:
                print("Error: Query cannot be empty.")
                return
            # ────────────────────────────────────────────────
            # Initialize hybrid search engine
            # ────────────────────────────────────────────────
            try:
                searcher = HybridSearch(documents)
                    # Add any required arguments your class needs, e.g.:
                    # documents_path="data/movies.json",
                    # embeddings_path="data/embeddings.pkl",
                    # etc.
            except Exception as e:
                print(f"Failed to initialize search engine: {e}")
                return
            # ────────────────────────────────────────────────
            # 1. Perform RRF search → get top results
            # ────────────────────────────────────────────────
            print(f"Searching for: {query!r} ...", flush=True)
            try:
                results = searcher.rrf_search(
                    query=query,
                    limit=args.top_k,
                    summarize=args.summarize,
                )
            except Exception as e:
                print(f"Search failed: {e}")
                return
            if not results:
                print("No relevant documents found.")
                return
            # ────────────────────────────────────────────────
            # 2. Format documents for prompt
            # ────────────────────────────────────────────────
            docs_text = format_documents_for_prompt(results)
            # ────────────────────────────────────────────────
            # 3. Build RAG prompt
            # ────────────────────────────────────────────────
            prompt = f"""Answer the question or provide information based on the provided documents. 
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Documents:
            {docs_text}
            Provide a comprehensive, natural and engaging answer that addresses the query.
            Use information only from the provided documents.
            If the documents do not contain enough information, say so clearly.
            Write in a friendly, approachable tone suitable for a streaming service user."""
            # ────────────────────────────────────────────────
            # 4. Call Gemini
            # ────────────────────────────────────────────────
            from google import genai
            from google.genai import types
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY not found in environment.")
                return
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",           # or "gemini-1.5-pro" / "gemini-2.5-flash"
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=600,
                    )
                )
                answer = response.text.strip()
            except Exception as e:
                print(f"Gemini API call failed: {e}")
                return
            # ────────────────────────────────────────────────
            # 5. Print formatted output
            # ────────────────────────────────────────────────
            print("\n" + "="*70)
            print("Search Results:")
            for i, doc in enumerate(results, 1):
                title = doc.get("title", "Untitled")
                print(f"  - {title}")
            print("\nRAG Response:")
            print(answer)
            print("="*70 + "\n")
        case 'summarize':
            # Create HybridSearch instance and perform RRF search
            hybrid = HybridSearch(documents)
            results = hybrid.rrf_search(query=args.query, k=60, limit=args.limit)
            # Format results
            results_text = "\n".join([
                f"- {r['title']}: {r['description']}"
                for r in results
            ])
            # Summarize with Gemini
            summary = generate_summary(args.query, results_text)
            # Print results
            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")
            print("\nLLM Summary:")
            print(summary)
        case 'citations':
            # Create HybridSearch instance and perform RRF search
            hybrid = HybridSearch(documents)
            results = hybrid.rrf_search(query=args.query, k=60, limit=args.limit)
            # Format results
            results_text = "\n".join([
                f"- {r['title']}: {r['description']}"
                for r in results
            ])
            # Summarize with Gemini and include citations
            summary = generate_citations(args.query, results_text)
            # Print results
            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")
            print("\nLLM Answer:")
            print(summary)
        case 'question':
            # Create HybridSearch instance and perform RRF search
            hybrid = HybridSearch(documents)
            results = hybrid.rrf_search(query=args.query, k=60, limit=args.limit)
            # Format results
            results_text = "\n".join([
                f"- {r['title']}: {r['description']}"
                for r in results
            ])
            # Summarize with Gemini and answer question
            summary = answer_question(args.query, results_text)
            # Print results
            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")
            print("\nAnswer:")
            print(summary)
        case _:
            parser.print_help()

def generate_summary(query: str, results_text: str) -> str:
    """Generate summary using Gemini."""
    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {results_text}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) 
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )
    return response.text.strip()

def generate_citations(query: str, results_text: str) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.
    Query: {query}
    Documents:
    {results_text}
    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative
    Answer:"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) 
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )
    return response.text.strip()

def answer_question(query: str, results_text: str) -> str:
    prompt = f"""Answer the following question based on the provided documents.
    Question: {query}
    Documents:
    {results_text}
    General instructions:
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible
    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view
    Answer:"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) 
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )
    return response.text.strip()

if __name__ == "__main__":
    main()