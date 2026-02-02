#!/usr/bin/env python3
"""
CLI tool to rewrite a text query using Gemini, combining it with visual information
from an image (optimized for movie-related search improvement).
"""
import os
import sys
import argparse
import mimetypes
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
import torch
from PIL import Image
from transformers import AutoModel

def main():
    parser = argparse.ArgumentParser(description="Rewrite movie query using image + text with Gemini")
    parser.add_argument("--image", required=True, type=str, help="Path to the image file (e.g. data/paddington.jpeg)")
    parser.add_argument("--query", required=True, type=str, help="Text query to rewrite based on the image content")
    args = parser.parse_args()
    # Resolve and validate image path
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print(f"Error: Image file not found: {image_path}")
        return 1
    # Determine MIME type
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/jpeg"
    # Read image as bytes
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        print(f"Error reading image: {e}")
        return 1
    # Configure Gemini API key (must be set in environment)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        return
    client = genai.Client(api_key=api_key)
    # System prompt (exact wording from assignment)
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""
    # Build content parts (new style)
    content_parts = [
        system_prompt,
        types.Part.from_bytes(
            data=img_bytes,
            mime_type=mime
        ),
        args.query.strip(),
    ]
    try:
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash",          # or "gemini-1.5-flash", "gemini-2.5-pro", etc.
            contents=content_parts,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=100,
            ),
        )
        # Output results
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        if hasattr(e, 'response'):
            print(e.response)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())