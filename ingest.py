#!/usr/bin/env python3
"""Ingest property data into ChromaDB vector store.

Usage:
    python ingest.py
    python ingest.py --data data/mohegan_sun.json --persist data/chroma
"""

import argparse
import logging

from src.rag.pipeline import ingest_property

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest property data into ChromaDB.")
    parser.add_argument(
        "--data",
        default="data/mohegan_sun.json",
        help="Path to property JSON file (default: data/mohegan_sun.json)",
    )
    parser.add_argument(
        "--persist",
        default="data/chroma",
        help="ChromaDB persist directory (default: data/chroma)",
    )
    args = parser.parse_args()

    vectorstore = ingest_property(data_path=args.data, persist_dir=args.persist)
    if vectorstore:
        # Quick verification
        results = vectorstore.similarity_search("restaurants", k=2)
        print(f"\nVerification: found {len(results)} results for 'restaurants'")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] ({doc.metadata.get('category', '?')}) {doc.page_content[:100]}...")
    else:
        print("No documents ingested. Check that the data file exists.")


if __name__ == "__main__":
    main()
