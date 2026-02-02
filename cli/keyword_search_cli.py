#!/usr/bin/env python3

import argparse
import math
from lib.text_processing import tokenize_and_stem
from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex
from lib.keyword_search import bm25_idf_command, bm25_tf_command
from lib.search_utils import BM25_K1, BM25_B


def main() -> None:
    # Define CLI subcommands and their arguments
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keyword search")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Search term")
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=None, help="Limit the number of results")
    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. [{res['id']}] {res['title']}")
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            docs = index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Error: inverted index not found. Run 'build' first.")
                return
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            tokens = tokenize_and_stem(args.term)
            if len(tokens) != 1:
                raise ValueError("IDF expects a single token")
            term = tokens[0]
            total_doc_count = len(index.docmap)
            matching_docs = index.get_documents(term)
            term_match_doc_count = len(matching_docs)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            # Tokenize and validate single term
            tokens = tokenize_and_stem(args.term)
            if len(tokens) != 1:
                raise ValueError("TF-IDF expects a single token")
            term = tokens[0]
            # TF
            tf = index.get_tf(args.doc_id, term)
            # IDF
            total_docs = len(index.docmap)
            matching_docs = len(index.index.get(term, set()))
            idf = math.log((total_docs + 1) / (matching_docs + 1))
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except FileNotFoundError:
                print("Error: inverted index not found. Run 'build' first.")
            except ValueError as e:
                print(f"Error: {e}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except FileNotFoundError:
                print("Error: inverted index not found. Run 'build' first.")
            except ValueError as e:
                print(f"Error: {e}")
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(
                    args.doc_id,
                    args.term,
                    args.k1,
                    args.b
                )
                print(
                    f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
                )
            except FileNotFoundError:
                print("Error: inverted index not found. Run 'build' first.")
            except ValueError as e:
                print(f"Error: {e}")
        case "bm25search":
            try:
                query = args.query
                print("BM25 Searching for:", query)
                index = InvertedIndex()
                index.load()
                query_tokens = tokenize_and_stem(query)
                results = index.bm25_search(query, limit=args.limit)
                for i, result in enumerate(results, start=1):
                    print(
                        print(f"{i}. ({result['id']}) {result['title']} - Score: {result['score']:.2f}")
                    )
            except FileNotFoundError as e:
                print("Error: inverted index not found. Run 'build' first.")
            except ValueError as e:
                print(f"Error: {e}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
