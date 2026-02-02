#!/usr/bin/env python3
import argparse
import re
from unittest import case
from lib.semantic_search import verify_embeddings, verify_model, embed_text, embed_query_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add verify subcommand
    subparsers.add_parser('verify', help='Verify the model is loaded correctly')
    
    # Add embed_text subcommand with a text argument
    embed_parser = subparsers.add_parser('embed_text', help='Embed a text string')
    embed_parser.add_argument('text', type=str, help='Text to embed')
    
    # Add verify_embeddings subcommand
    subparsers.add_parser('verify_embeddings', help='Verify embeddings are loaded correctly')
    
    # Add embedquery subcommand with a text argument
    query_parser = subparsers.add_parser('embedquery', help='Embed a query string')
    query_parser.add_argument('text', type=str, help='Text to query embed')

    # Add search command with query and limit arguments
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', type=str, help='Query to search for')
    search_parser.add_argument('--limit', type=int, default=5, help='Limit the number of results')

    # Add chunk command with text argument
    chunk_parser = subparsers.add_parser('chunk', help='Chunk a text string')
    chunk_parser.add_argument('text', type=str, help='Text to chunk')
    chunk_parser.add_argument('--chunk-size', type=int, default=200, help='Size of each chunk')
    chunk_parser.add_argument('--overlap', type=int, default=0, help='Size of overlap between chunks')
    
    # Add semantic_chunk subcommand
    semantic_chunk_parser = subparsers.add_parser('semantic_chunk', help='Chunk text by sentences')
    semantic_chunk_parser.add_argument('text', type=str, help='Text to chunk')
    semantic_chunk_parser.add_argument('--max-chunk-size', type=int, default=4, help='Maximum sentences per chunk')
    semantic_chunk_parser.add_argument('--overlap', type=int, default=0, help='Number of overlapping sentences')

    # Add embed_chunks subcommand
    subparsers.add_parser('embed_chunks', help='Generate embeddings for chunked documents')

    # Add search_chunked command
    search_chunked_parser = subparsers.add_parser('search_chunked', help='Search for a chunked document')
    search_chunked_parser.add_argument('query', type=str, help='Query to search for')
    search_chunked_parser.add_argument('--limit', type=int, default=5, help='Limit the number of results')

    # Parse arguments to pass to the appropriate function (match/case)
    args = parser.parse_args()

    match args.command:
        case 'verify':
            verify_model()
        case 'embed_text':
            #text = input("Enter text to embed: ")
            embed_text(args.text)
        case 'verify_embeddings':
            verify_embeddings()
        case 'embedquery':
            embed_query_text(args.text)
        case 'search':
            from lib.semantic_search import SemanticSearch
            import json
            # Create search instance and load embeddings
            search_engine = SemanticSearch()
            with open('data/movies.json', 'r') as f:
                data = json.load(f)
            documents = data['movies']
            search_engine.load_or_create_embeddings(documents)
            # Perform search
            results = search_engine.search(args.query, limit=args.limit)
            # Print results
            for i, result in enumerate(results, start=1):
                # Truncate description to first 100 characters
                description = result['description'][:100] + '...'
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {description}")   
        case 'chunk':
            words = args.text.split()
            chunks = []
            i = 0      
            while i < len(words):
                # Get chunk starting at position i
                chunk = ' '.join(words[i:i + args.chunk_size])
                chunks.append(chunk)        
                # Move forward by chunk_size minus overlap
                i += args.chunk_size - args.overlap
            # Print results
            total_chars = len(args.text)
            print(f"Chunking {total_chars} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        case 'semantic_chunk':
            from lib.semantic_search import semantic_chunk
            # Use the semantic_chunk function (it handles all the logic)
            chunks = semantic_chunk(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
            # Print results
            total_chars = len(args.text)
            print(f"Semantically chunking {total_chars} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        case 'embed_chunks':
            from lib.semantic_search import ChunkedSemanticSearch
            from lib.search_utils import load_movies
            movies = load_movies()
            chunked_search = ChunkedSemanticSearch()
            embeddings = chunked_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case 'search_chunked':
            from lib.semantic_search import ChunkedSemanticSearch
            from lib.search_utils import load_movies
            movies = load_movies()
            chunked_search = ChunkedSemanticSearch()
            chunked_search.load_or_create_chunk_embeddings(movies)
            results = chunked_search.search_chunks(args.query, limit=args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document'][:100]}...")

        case _:
            parser.print_help() 

if __name__ == "__main__":
    main()