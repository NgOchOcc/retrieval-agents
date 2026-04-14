"""Helper script to download and prepare Wikipedia corpus."""

import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset


def download_wiki_dpr(cache_dir="./cache", max_passages=None):
    """
    Download DPR Wikipedia corpus (pre-chunked passages).

    Args:
        cache_dir: Directory to save the corpus
        max_passages: Maximum number of passages to download (None for all)
    """
    print("Downloading DPR Wikipedia corpus...")
    print("This corpus contains ~21M passages from Wikipedia")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "wikipedia_paragraphs.json")

    if os.path.exists(cache_file):
        print(f"Wikipedia corpus already exists at {cache_file}")
        with open(cache_file, "r") as f:
            data = json.load(f)
        print(f"Found {len(data)} passages")
        return

    # Load DPR Wikipedia passages
    try:
        dataset = load_dataset("facebook/dpr-ctx_encoder-multiset-base", split="train")
        print(f"Loaded {len(dataset)} passages")
    except Exception as e:
        print(f"Error loading facebook/dpr dataset: {e}")
        print("Trying alternative: wiki_dpr...")
        try:
            dataset = load_dataset("wiki_dpr", "psgs_w100.multiset.no_index", split="train")
            print(f"Loaded {len(dataset)} passages")
        except Exception as e2:
            print(f"Error: {e2}")
            print("\nFalling back to simple Wikipedia articles...")
            download_wikipedia_articles(cache_dir, max_articles=10000 if max_passages else None)
            return

    # Limit passages if specified
    if max_passages and max_passages < len(dataset):
        dataset = dataset.select(range(max_passages))
        print(f"Limited to {max_passages} passages for testing")

    # Convert to our format
    paragraphs = []
    for i, item in enumerate(tqdm(dataset, desc="Processing passages")):
        title = item.get("title", "")
        text = item.get("text", "")

        if text.strip():
            para = {
                "para_id": f"wiki_{i}",
                "title": title,
                "text": text.strip(),
                "sentence_id": 0
            }
            paragraphs.append(para)

    # Save to cache
    print(f"Saving {len(paragraphs)} passages to {cache_file}")
    with open(cache_file, "w") as f:
        json.dump(paragraphs, f)

    print("Done!")


def download_wikipedia_articles(cache_dir="./cache", max_articles=None):
    """
    Download Wikipedia articles and split into sentences.

    Args:
        cache_dir: Directory to save the corpus
        max_articles: Maximum number of articles to process (None for all)
    """
    print("Downloading Wikipedia articles...")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "wikipedia_paragraphs.json")

    if os.path.exists(cache_file):
        print(f"Wikipedia corpus already exists at {cache_file}")
        return

    # Try loading Wikipedia dataset
    try:
        print("Trying wikimedia/wikipedia dataset...")
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    except Exception as e:
        print(f"Error loading wikimedia/wikipedia: {e}")
        print("\nPlease manually download Wikipedia corpus or use a smaller dataset.")
        return

    # Limit articles if specified
    if max_articles and max_articles < len(dataset):
        dataset = dataset.select(range(max_articles))
        print(f"Limited to {max_articles} articles")

    # Process articles
    paragraphs = []
    para_counter = 0

    for article in tqdm(dataset, desc="Processing articles"):
        title = article.get("title", "")
        text = article.get("text", "")

        # Simple sentence splitting
        sentences = split_into_sentences(text)

        for sent_id, sentence in enumerate(sentences):
            if sentence.strip():
                para = {
                    "para_id": f"wiki_{para_counter}",
                    "title": title,
                    "text": sentence.strip(),
                    "sentence_id": sent_id
                }
                paragraphs.append(para)
                para_counter += 1

    # Save to cache
    print(f"Saving {len(paragraphs)} sentences to {cache_file}")
    with open(cache_file, "w") as f:
        json.dump(paragraphs, f)

    print("Done!")


def split_into_sentences(text):
    """Simple sentence splitting."""
    sentences = []
    for line in text.split("\n"):
        if line.strip():
            # Split by period followed by space
            parts = line.split(". ")
            for part in parts:
                if part.strip():
                    sentences.append(part.strip())
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia corpus for HotpotQA benchmarking")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--max_passages", type=int, default=None, help="Max passages to download (for testing)")
    parser.add_argument("--use_articles", action="store_true", help="Use Wikipedia articles instead of DPR passages")
    parser.add_argument("--max_articles", type=int, default=None, help="Max articles to process")

    args = parser.parse_args()

    if args.use_articles:
        download_wikipedia_articles(args.cache_dir, args.max_articles)
    else:
        download_wiki_dpr(args.cache_dir, args.max_passages)


if __name__ == "__main__":
    main()
