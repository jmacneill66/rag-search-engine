import argparse
import json
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit
    # Load golden dataset
    with open("data/golden_dataset.json", "r") as f:
        data = json.load(f)
    # Extract the test cases list
    golden_dataset = data["test_cases"]
    # Load movies and initialize hybrid search
    documents = load_movies()
    hybrid = HybridSearch(documents)
    print(f"k={limit}\n")
    # Run evaluation for each test case
    for test_case in golden_dataset:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        # Run RRF search (k=60 is the RRF constant, limit is top-k results)
        results = hybrid.rrf_search(
            query=query,
            k=60,
            limit=limit
        )
        # Get retrieved titles
        retrieved_titles = [r["title"] for r in results]
        # Calculate precision@k
        # (number of relevant titles found in results) / (number of results)
        relevant_found = sum(
            1 for title in retrieved_titles if title in relevant_titles
        )
        precision = relevant_found / limit if limit > 0 else 0.0
        # Calculate recall@k   
        # (number of relevant docs found in results) / (total number of relevant docs)
        total_relevant = len(relevant_titles)
        recall = relevant_found / total_relevant if total_relevant > 0 else 0.0
        # Calculate F1@k - the harmonic mean of precision and recall  
        #  F1 score – a single metric that balances precision and recall when both are equally important
        f1 = 2 * (precision * recall) / (precision + recall)
        # Print results
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()

if __name__ == "__main__":
    main()