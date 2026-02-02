#!/usr/bin/env python3
"""
CLI for multimodal (CLIP-based) image & movie search utilities.
"""
import argparse
import sys
from pathlib import Path
from lib.multimodal_search import (verify_image_embedding, image_search_command)

def print_search_results(results):
    if not results:
        print("No results found.")
        return
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (similarity: {res['similarity']})")
        print(f"   {res['description'][:150]}...")  # truncate long descriptions
        print()


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI (CLIP embeddings)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify-image-embedding", help="Generate CLIP embedding and print its dimensionality",)
    verify_parser.add_argument("image_path", type=str, help="Path to the image file",)
    search_parser = subparsers.add_parser("image_search", help="Search movies using an uploaded image (CLIP-based)",)
    search_parser.add_argument("image_path", type=str, help="Path to the query image (e.g. data/paddington.jpeg)",)
    args = parser.parse_args()
    if args.command == "verify-image-embedding":
        image_path = Path(args.image_path).expanduser().resolve()
        if not image_path.is_file():
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            return 1
        try:
            verify_image_embedding(image_path)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.command == "image_search":
        image_path = Path(args.image_path).expanduser().resolve()
        if not image_path.is_file():
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            return 1
        try:
            results = image_search_command(image_path)
            print_search_results(results)
        except Exception as e:
            print(f"Error during image search: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())