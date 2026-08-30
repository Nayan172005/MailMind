"""
MailMind RAG — CLI Demo

Interactive command-line interface for testing the RAG pipeline.

Usage:
    python -m rag.run                  # Interactive mode
    python -m rag.run --build-index    # Build FAISS index first
"""

import argparse
import sys
import uuid

from rag.pipeline import RAGPipeline
from rag.utils import get_logger

logger = get_logger("cli")


def print_separator():
    print("\n" + "=" * 60)


def print_results(results):
    """Print search results."""
    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r.subject or '(No Subject)'}")
        print(f"     From: {r.sender_name or r.sender_email}")
        print(f"     Date: {r.date}")
        print(f"     Score: {r.score}")
        if r.snippet:
            snippet = r.snippet[:150] + "..." if len(r.snippet) > 150 else r.snippet
            print(f"     Preview: {snippet}")


def run_interactive(pipeline: RAGPipeline):
    """Run the interactive CLI loop."""
    session_id = str(uuid.uuid4())[:8]

    print_separator()
    print("  MailMind — Intelligent Email Assistant")
    print("  Type your question or command below.")
    print()
    print("  Commands:")
    print("    /search <query>      — Semantic email search")
    print("    /ask <query>         — Question answering")
    print("    /summarize <query>   — Summarize matching emails")
    print("    /summary <msg_id>    — Summarize a specific email")
    print("    /conversation        — Toggle conversation mode")
    print("    /clear               — Clear conversation history")
    print("    /status              — Show index status")
    print("    /quit                — Exit")
    print()
    print("  Or just type a question directly (uses /ask mode).")
    print_separator()

    conversation_mode = False

    while True:
        try:
            prompt = "\n[conv] > " if conversation_mode else "\n> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        elif user_input.lower() == "/status":
            status = pipeline.get_index_status()
            print(f"\n  Index Status:")
            print(f"    Indexed: {status.indexed}")
            print(f"    Vectors: {status.total_vectors}")
            print(f"    Emails:  {status.total_emails}")
            print(f"    Path:    {status.index_path}")
            continue

        elif user_input.lower() == "/conversation":
            conversation_mode = not conversation_mode
            mode_str = "ON" if conversation_mode else "OFF"
            print(f"\n  Conversation mode: {mode_str}")
            if not conversation_mode:
                pipeline.clear_conversation(session_id)
            continue

        elif user_input.lower() == "/clear":
            pipeline.clear_conversation(session_id)
            print("\n  Conversation history cleared.")
            continue

        elif user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            if not query:
                print("  Please provide a search query.")
                continue

            print(f"\n  Searching: \"{query}\"")
            try:
                response = pipeline.search(query)
                print(f"\n  Found {response.total} results:")
                print_results(response.results)
            except Exception as e:
                print(f"\n  Error: {e}")
            continue

        elif user_input.lower().startswith("/summarize "):
            query = user_input[11:].strip()
            if not query:
                print("  Please provide a search query for summarization.")
                continue

            print(f"\n  Summarizing emails about: \"{query}\"")
            try:
                response = pipeline.summarize(query=query)
                print(f"\n  Summary:\n  {response.summary}")
                if response.sources:
                    print(f"\n  Based on {len(response.sources)} email(s):")
                    print_results(response.sources)
            except Exception as e:
                print(f"\n  Error: {e}")
            continue

        elif user_input.lower().startswith("/summary "):
            msg_id = user_input[9:].strip()
            if not msg_id:
                print("  Please provide a message ID.")
                continue

            print(f"\n  Summarizing email: {msg_id}")
            try:
                response = pipeline.summarize(message_id=msg_id)
                print(f"\n  Summary:\n  {response.summary}")
            except Exception as e:
                print(f"\n  Error: {e}")
            continue

        elif user_input.lower().startswith("/ask "):
            query = user_input[5:].strip()
        else:
            # Default: treat as /ask or conversation
            query = user_input

        if not query:
            print("  Please provide a question.")
            continue

        try:
            if conversation_mode:
                print(f"\n  Thinking (conversation)...")
                response = pipeline.conversation_ask(
                    session_id=session_id,
                    query=query,
                )
                print(f"\n  Answer:\n  {response.answer}")
                if response.sources:
                    print(f"\n  Sources ({len(response.sources)}):")
                    print_results(response.sources)
            else:
                print(f"\n  Thinking...")
                response = pipeline.ask(query)
                print(f"\n  Answer:\n  {response.answer}")
                if response.sources:
                    print(f"\n  Sources ({len(response.sources)}):")
                    print_results(response.sources)
        except Exception as e:
            print(f"\n  Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="MailMind RAG — CLI Demo",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the FAISS index before starting interactive mode.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the FAISS index even if it exists.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the email dataset (overrides config).",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.build_index or args.rebuild:
        print("\n  Building FAISS index...")
        try:
            status = pipeline.build_index(
                dataset_path=args.dataset,
                force_rebuild=args.rebuild,
            )
            print(f"  Index built: {status.total_vectors} vectors, "
                  f"{status.total_emails} emails")
        except Exception as e:
            print(f"  Error building index: {e}")
            sys.exit(1)
    else:
        # Try to load existing index
        try:
            pipeline.load_index()
        except FileNotFoundError:
            print("\n  No FAISS index found. Building one now...")
            try:
                status = pipeline.build_index()
                print(f"  Index built: {status.total_vectors} vectors, "
                      f"{status.total_emails} emails")
            except Exception as e:
                print(f"  Error building index: {e}")
                sys.exit(1)

    run_interactive(pipeline)


if __name__ == "__main__":
    main()
