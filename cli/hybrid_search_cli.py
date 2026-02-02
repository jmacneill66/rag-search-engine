import argparse
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize subcommand (existing)
    normalize_parser = subparsers.add_parser('normalize', help='Normalize movie scores')
    normalize_parser.add_argument('scores', type=float, nargs="+", help='Movie scores to normalize')

    # weighted-search subcommand (existing)
    weighted_parser = subparsers.add_parser('weighted-search', help='Perform weighted hybrid search')
    weighted_parser.add_argument('query', type=str, help='Search query')
    weighted_parser.add_argument('--alpha', type=float, default=0.5, 
                                 help='Weight for BM25 score (0.0 = pure semantic, 1.0 = pure BM25)')
    weighted_parser.add_argument('--limit', type=int, default=5, 
                                 help='Number of results to return (default: 5)')
    # rrf-search subcommand 
    rrf_parser = subparsers.add_parser('rrf-search', help='Perform RRF hybrid search')
    rrf_parser.add_argument('query', type=str, help='Search query')
    rrf_parser.add_argument('-k', type=int, default=60, 
                            help='RRF constant (default: 60)')
    rrf_parser.add_argument('--limit', type=int, default=5, 
                            help='Number of results to return (default: 5)')
    rrf_parser.add_argument("--enhance",
                            type=str,
                            choices=["spell", "rewrite", "expand"],
                            help="Query enhancement method",)   
    rrf_parser.add_argument("--rerank-method",
                            type=str,
                            choices=["individual", "batch", "cross_encoder"],
                            help="Re-ranking method to apply after initial search",)
    rrf_parser.add_argument("--evaluate", action="store_true",
                            help="Evaluate results using LLM relevance scoring (0-3)")               
    args = parser.parse_args()

    from lib.search_utils import load_movies
    documents = load_movies()
    hybrid = HybridSearch(documents)
    match args.command:
        case 'normalize':
            HybridSearch.normalize(args.scores)
        case 'weighted-search':
            results = hybrid.weighted_search(
                query=args.query,
                alpha=args.alpha,
                limit=args.limit
            )     
            for i, result in enumerate(results, 1):
                title = result["title"]
                description = result["description"][:100]
                if len(result["description"]) > 100:
                    description += "..."   
                print(f"{i}. {title}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_norm']:.3f}, Semantic: {result['semantic_norm']:.3f}")
                print(f"   {description}")
                print()

        case 'rrf-search':  #Reciprocal Rank Fusion
            original_query = args.query
            enhanced_query = original_query
            # Apply query enhancement if requested
            if args.enhance == "spell":
                from lib.hybrid_search import enhance_query_with_spell_correction
                enhanced_query = enhance_query_with_spell_correction(original_query)
                # Print enhancement info if query changed
                if enhanced_query != original_query:
                    print(f"Enhanced query (spell): '{original_query}' -> '{enhanced_query}'\n")
            elif args.enhance == "rewrite":
                from lib.hybrid_search import enhance_query_with_rewrite
                enhanced_query = enhance_query_with_rewrite(original_query)
                # Print enhancement info if query changed
                if enhanced_query != original_query:
                    print(f"Enhanced query (rewrite): '{original_query}' -> '{enhanced_query}'\n")
            elif args.enhance == "expand":
                from lib.hybrid_search import enhance_query_with_expand
                enhanced_query = enhance_query_with_expand(original_query)
                # Print enhancement info if query changed
                if enhanced_query != original_query:
                    print(f"Enhanced query (expand): '{original_query}' -> '{enhanced_query}'\n")
            # Determine how many results to fetch
            fetch_limit = args.limit
            if args.rerank_method:
                fetch_limit = args.limit * 5
            # Use enhanced query for search
            results = hybrid.rrf_search(
                query=enhanced_query,
                k=args.k,
                limit=fetch_limit
                )
            # Apply re-ranking if requested
            if args.rerank_method == "individual":
                from lib.hybrid_search import rerank_individual
                print(f"Reranking top {fetch_limit} results using individual method...")
                results = rerank_individual(enhanced_query, results)
                # Truncate to original limit after re-ranking
                results = results[:args.limit]
            elif args.rerank_method == "batch":
                from lib.hybrid_search import rerank_batch
                print(f"Reranking top {fetch_limit} results using batch method...\n")
                results = rerank_batch(enhanced_query, results)
                results = results[:args.limit]
            elif args.rerank_method == "cross_encoder":
                from lib.hybrid_search import rerank_cross_encoder
                print(f"Reranking top {fetch_limit} results using cross_encoder method...\n")
                results = rerank_cross_encoder(enhanced_query, results)
                results = results[:args.limit]
            # Print header
            print(f"Reciprocal Rank Fusion Results for '{enhanced_query}' (k={args.k}):\n")
            # Print results
            for i, result in enumerate(results, 1):
                title = result["title"]
                desc_snippet = result["description"][:100]
                if len(result["description"]) > 100:
                    desc_snippet += "..."
                bm25_str = str(result["bm25_rank"]) if result["bm25_rank"] is not None else "—"
                sem_str = str(result["semantic_rank"]) if result["semantic_rank"] is not None else "—"
                print(f"{i}. {title}")
                # Print rerank score if present
                if "rerank_score" in result:
                    print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
                if "rerank_rank" in result:
                    print(f"   Rerank Rank: {result['rerank_rank']}")
                if "cross_encoder_score" in result:
                    print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_str}, Semantic Rank: {sem_str}")
                print(f"   {desc_snippet}")
                print()
            if args.evaluate:
                from lib.hybrid_search import evaluate_with_llm   
                evaluate_with_llm(args.query, results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()