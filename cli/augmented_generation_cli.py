import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.augmented_utils.gemini import generate_response, generate_summary, generate_citations, answer_question
from utils.augmented_utils.rrf_search import do_rrf_search

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize content")
    summarize_parser.add_argument("query", type=str, help="Search query for summary")
    summarize_parser.add_argument("--limit", type=int, default=5, help="limit of documents")

    citation_parser = subparsers.add_parser("citations", help="Generate citations")
    citation_parser.add_argument("query", type=str, help="Search query for citations")
    citation_parser.add_argument("--limit", type=int, default=5, help="limit of documents")

    question_parser = subparsers.add_parser("question", help="Answer a question")
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="limit of documents")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            cmd = [
                "uv",
                "run",
                "cli/hybrid_search_cli.py",
                "rrf_search",
                query,
                "--k=60",
                "--limit=5",
                "--json",
            ]
            
            search_results = do_rrf_search(cmd)

            print("Search Results:")
            for item in search_results:
                title = item.get("title", "Untitled")
                print(f"  - {title}")

            response = generate_response(query, search_results)
            print("\nRAG Response:")
            print(response)

        case "summarize":
            query = args.query
            limit = args.limit

            cmd = [
                "uv",
                "run",
                "cli/hybrid_search_cli.py",
                "rrf_search",
                query,
                "--k=60",
                f"--limit={limit}",
                "--json",
            ]
            search_results = do_rrf_search(cmd)

            print("Search Results:")
            for item in search_results:
                title = item.get("title", "Untitled")
                print(f"  - {title}")

            response = generate_summary(query, search_results)
            print("LLM Summary:")
            print(response)

        case "citations":
            query = args.query
            limit = args.limit

            cmd = [
                "uv",
                "run",
                "cli/hybrid_search_cli.py",
                "rrf_search",
                query,
                "--k=60",
                f"--limit={limit}",
                "--json",
            ]
            search_results = do_rrf_search(cmd)

            print("Search Results:")
            for item in search_results:
                title = item.get("title", "Untitled")
                print(f"  - {title}")

            response = generate_citations(query, search_results)
            print("LLM Answer:")
            print(response)

        case "question":
            question = args.question
            limit = args.limit

            cmd = [
                "uv",
                "run",
                "cli/hybrid_search_cli.py",
                "rrf_search",
                question,
                "--k=60",
                f"--limit={limit}",
                "--json",
            ]
            search_results = do_rrf_search(cmd)

            print("Search Results:")
            for item in search_results:
                title = item.get("title", "Untitled")
                print(f"  - {title}")

            response = answer_question(question, search_results)
            print("Answer:")
            print(response)
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
