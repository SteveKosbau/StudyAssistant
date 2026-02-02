#!/usr/bin/env python3
"""
CLI Query Script for Study Assistant
Quick terminal-based lookups with citations.
"""

import os
import sys
import argparse

import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# Configuration
CHROMA_PATH = os.path.expanduser("~/Desktop/StudyAssistant/chroma_db")
COLLECTION_NAME = "study_materials"
TOP_K = 5  # Number of chunks to retrieve


def get_relevant_chunks(query: str, top_k: int = TOP_K) -> list:
    """Retrieve relevant chunks from ChromaDB."""
    # Initialize embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        print("Error: No documents found. Please run ingest.py first.")
        sys.exit(1)

    if collection.count() == 0:
        print("Error: Database is empty. Please run ingest.py first.")
        sys.exit(1)

    # Generate query embedding
    query_embedding = embedder.encode(query).tolist()

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results['ids'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "filename": results['metadatas'][0][i]['filename'],
            "pages": results['metadatas'][0][i]['pages'],
            "distance": results['distances'][0][i]
        })

    return chunks


def query_claude(question: str, chunks: list) -> str:
    """Send question and context to Claude, get answer with citations."""

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']}, Page(s): {chunk['pages']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # System prompt enforcing citation requirements
    system_prompt = """You are a study assistant that answers questions ONLY based on the provided context from course materials.

CRITICAL RULES:
1. ONLY use information from the provided context to answer questions
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided materials to answer this question."
3. ALWAYS cite your sources using the format: [Source: filename, Page(s): X]
4. Include citations inline with your answer, immediately after the relevant information
5. If information comes from multiple sources, cite each one
6. Be precise and educational in your explanations
7. For visual analytics topics, describe concepts clearly even without the actual visuals

FORMAT YOUR RESPONSE:
- Start with a direct answer
- Provide explanation with inline citations
- End with a "Sources Used" summary listing all cited materials"""

    user_message = f"""Based on the following excerpts from my course materials, please answer my question.

CONTEXT FROM COURSE MATERIALS:
{context}

MY QUESTION: {question}

Remember: Only use information from the context above. Include citations for all information."""

    # Call Claude API
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(
        description="Query your study materials with AI-powered search and citations"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Your question (or use -i for interactive mode)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode - ask multiple questions"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of relevant chunks to retrieve (default: {TOP_K})"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show the retrieved source chunks before the answer"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please add it to your ~/.zshrc file.")
        sys.exit(1)

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print("Error: No database found. Please run ingest.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Study Assistant - Query Mode")
    print("=" * 60)

    # Load models once for interactive mode
    print("\nLoading models...")

    if args.interactive:
        print("\nInteractive mode. Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                question = input("\n📚 Your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not question:
                continue

            print("\n🔍 Searching course materials...")
            chunks = get_relevant_chunks(question, args.top_k)

            if args.show_sources:
                print("\n📄 Retrieved sources:")
                for i, chunk in enumerate(chunks, 1):
                    print(f"\n  [{i}] {chunk['filename']} (p. {chunk['pages']})")
                    preview = chunk['text'][:200].replace('\n', ' ')
                    print(f"      {preview}...")

            print("\n🤖 Generating answer...\n")
            answer = query_claude(question, chunks)
            print("-" * 60)
            print(answer)
            print("-" * 60)

    else:
        if not args.question:
            parser.print_help()
            print("\nExample: python query.py \"What is a treemap?\"")
            sys.exit(1)

        print(f"\n🔍 Searching for: {args.question}")
        chunks = get_relevant_chunks(args.question, args.top_k)

        if args.show_sources:
            print("\n📄 Retrieved sources:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n  [{i}] {chunk['filename']} (p. {chunk['pages']})")
                preview = chunk['text'][:200].replace('\n', ' ')
                print(f"      {preview}...")

        print("\n🤖 Generating answer...\n")
        answer = query_claude(args.question, chunks)
        print("-" * 60)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
