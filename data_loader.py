"""Data loading and preprocessing for HotpotQA."""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class HotpotQAExample:
    """HotpotQA example structure."""

    question_id: str
    question: str
    answer: str
    supporting_facts: List[Tuple[str, int]]  # (title, sentence_id)
    context: List[Tuple[str, List[str]]]  # (title, sentences)
    type: str  # 'comparison' or 'bridge'
    level: str  # 'hard', 'medium', 'easy'


@dataclass
class Paragraph:
    """Paragraph representation."""

    para_id: str
    title: str
    text: str
    sentence_id: int  # sentence index within the document


class HotpotQADataLoader:
    """Load and preprocess HotpotQA dataset."""

    def __init__(self, split: str = "test", config: str = "fullwiki", cache_dir: str = "./cache"):
        """
        Initialize HotpotQA data loader.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            config: Dataset configuration ('distractor' or 'fullwiki')
            cache_dir: Directory for caching data
        """
        self.split = split
        self.config = config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_dataset(self, max_samples: Optional[int] = None) -> List[HotpotQAExample]:
        """
        Load HotpotQA dataset from HuggingFace.

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            List of HotpotQA examples
        """
        print(f"Loading HotpotQA dataset: {self.config}/{self.split}")

        # Load from HuggingFace
        dataset = load_dataset("hotpot_qa", self.config, split=self.split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        examples = []
        for item in tqdm(dataset, desc="Processing examples"):
            example = HotpotQAExample(
                question_id=item["id"],
                question=item["question"],
                answer=item["answer"],
                supporting_facts=list(zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])),
                context=list(zip(item["context"]["title"], item["context"]["sentences"])),
                type=item["type"],
                level=item["level"],
            )
            examples.append(example)

        print(f"Loaded {len(examples)} examples")
        return examples

    def build_corpus_from_examples(self, examples: List[HotpotQAExample]) -> Tuple[List[Paragraph], Dict]:
        """
        Build paragraph corpus from HotpotQA examples (for distractor mode).

        Args:
            examples: List of HotpotQA examples

        Returns:
            Tuple of (paragraphs, supporting_facts_map)
            - paragraphs: List of all unique paragraphs
            - supporting_facts_map: Dict mapping question_id to set of paragraph IDs
        """
        print("Building paragraph corpus from examples...")

        paragraphs = []
        paragraph_dict = {}  # (title, sent_id) -> para_id
        supporting_facts_map = defaultdict(set)

        para_counter = 0

        for example in tqdm(examples, desc="Extracting paragraphs"):
            # Create set of supporting facts for this question
            supporting_set = set(example.supporting_facts)

            # Process all context paragraphs
            for title, sentences in example.context:
                for sent_id, sentence in enumerate(sentences):
                    key = (title, sent_id)

                    # Skip if already processed
                    if key in paragraph_dict:
                        para_id = paragraph_dict[key]
                    else:
                        # Create new paragraph
                        para_id = f"para_{para_counter}"
                        paragraph = Paragraph(
                            para_id=para_id, title=title, text=sentence, sentence_id=sent_id
                        )
                        paragraphs.append(paragraph)
                        paragraph_dict[key] = para_id
                        para_counter += 1

                    # Check if this is a supporting fact
                    if (title, sent_id) in supporting_set:
                        supporting_facts_map[example.question_id].add(para_id)

        print(f"Built corpus with {len(paragraphs)} unique paragraphs")
        return paragraphs, dict(supporting_facts_map)

    def load_wikipedia_corpus(self) -> List[Paragraph]:
        """
        Load full Wikipedia corpus for fullwiki evaluation.

        Returns:
            List of paragraphs from Wikipedia
        """
        print("Loading Wikipedia corpus...")

        # For fullwiki, we need the complete Wikipedia dump
        # HotpotQA fullwiki uses Wikipedia dump from 2017-10-01
        # We'll load the psgs_w100.tsv format used in DPR

        cache_file = os.path.join(self.cache_dir, "wikipedia_paragraphs.json")

        if os.path.exists(cache_file):
            print(f"Loading cached Wikipedia corpus from {cache_file}")
            with open(cache_file, "r") as f:
                data = json.load(f)
                paragraphs = [
                    Paragraph(
                        para_id=p["para_id"], title=p["title"], text=p["text"], sentence_id=p["sentence_id"]
                    )
                    for p in data
                ]
            print(f"Loaded {len(paragraphs)} paragraphs from cache")
            return paragraphs

        # Load Wikipedia from HuggingFace
        print("Downloading Wikipedia dataset from HuggingFace...")
        print("Note: This will download data on first run...")
        print("\nTIP: For faster setup, run: python download_wikipedia.py --max_passages 100000")
        print("     This will pre-download a smaller corpus for testing.\n")

        use_dpr_format = False
        wiki_dataset = None

        # Try multiple sources in order of preference
        sources = [
            ("facebook/dpr-ctx_encoder-multiset-base", True, "DPR passages"),
            ("wiki_dpr", True, "wiki_dpr passages"),
            ("wikimedia/wikipedia", False, "Wikipedia articles"),
        ]

        for source, is_dpr, description in sources:
            try:
                print(f"Attempting to load {description} from '{source}'...")
                if source == "wiki_dpr":
                    wiki_dataset = load_dataset(source, "psgs_w100.multiset.no_index", split="train")
                elif source == "wikimedia/wikipedia":
                    wiki_dataset = load_dataset(source, "20231101.en", split="train")
                else:
                    wiki_dataset = load_dataset(source, split="train")

                use_dpr_format = is_dpr
                print(f"Successfully loaded {description}")
                break
            except Exception as e:
                print(f"Failed to load from {source}: {e}")
                continue

        if wiki_dataset is None:
            raise RuntimeError(
                "Could not load Wikipedia corpus from any source. "
                "Please run 'python download_wikipedia.py' first or check your internet connection."
            )

        paragraphs = []
        para_counter = 0

        if use_dpr_format:
            # DPR format: already chunked into passages
            print("Using DPR pre-chunked passages...")
            for doc in tqdm(wiki_dataset, desc="Processing Wikipedia passages"):
                title = doc.get("title", "")
                text = doc.get("text", "")

                if text.strip():
                    para_id = f"wiki_{para_counter}"
                    # DPR passages are already at good granularity
                    paragraph = Paragraph(para_id=para_id, title=title, text=text.strip(), sentence_id=0)
                    paragraphs.append(paragraph)
                    para_counter += 1
        else:
            # Standard Wikipedia format: need to chunk into sentences
            print("Processing Wikipedia articles into sentences...")
            for doc in tqdm(wiki_dataset, desc="Processing Wikipedia articles"):
                title = doc["title"]
                text = doc["text"]

                # Split into sentences (simple split by periods)
                sentences = self._split_into_sentences(text)

                for sent_id, sentence in enumerate(sentences):
                    if sentence.strip():  # Skip empty sentences
                        para_id = f"wiki_{para_counter}"
                        paragraph = Paragraph(para_id=para_id, title=title, text=sentence.strip(), sentence_id=sent_id)
                        paragraphs.append(paragraph)
                        para_counter += 1

        print(f"Processed {len(paragraphs)} paragraphs from Wikipedia")

        # Cache the corpus
        print(f"Caching corpus to {cache_file}")
        cache_data = [
            {"para_id": p.para_id, "title": p.title, "text": p.text, "sentence_id": p.sentence_id}
            for p in paragraphs
        ]
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        return paragraphs

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple split by newlines and periods
        # For production, consider using nltk or spacy
        sentences = []
        for line in text.split("\n"):
            if line.strip():
                # Split by period followed by space
                parts = line.split(". ")
                for part in parts:
                    if part.strip():
                        sentences.append(part.strip())
        return sentences

    def get_ground_truth_labels(
        self, examples: List[HotpotQAExample], paragraphs: List[Paragraph]
    ) -> Dict[str, Dict[str, int]]:
        """
        Create ground truth relevance labels for evaluation.

        Args:
            examples: List of HotpotQA examples
            paragraphs: List of all paragraphs in corpus

        Returns:
            Dict mapping question_id -> {para_id: relevance_label}
            relevance_label: 1 for supporting facts, 0 otherwise
        """
        print("Creating ground truth labels...")

        # Create mapping from (title, sent_id) to para_id
        title_sent_to_para = {}
        for para in paragraphs:
            key = (para.title, para.sentence_id)
            title_sent_to_para[key] = para.para_id

        # Debug: Check mapping
        print(f"Total paragraphs in corpus: {len(paragraphs)}")
        print(f"Unique (title, sent_id) keys: {len(title_sent_to_para)}")

        ground_truth = {}
        total_supporting_facts = 0
        found_supporting_facts = 0

        for example in tqdm(examples, desc="Building ground truth"):
            labels = {}

            # Mark supporting facts as relevant (label=1)
            for title, sent_id in example.supporting_facts:
                total_supporting_facts += 1
                key = (title, sent_id)
                if key in title_sent_to_para:
                    para_id = title_sent_to_para[key]
                    labels[para_id] = 1
                    found_supporting_facts += 1

            ground_truth[example.question_id] = labels

        print(f"Created ground truth for {len(ground_truth)} questions")
        print(f"Total supporting facts: {total_supporting_facts}")
        print(f"Found in corpus: {found_supporting_facts} ({found_supporting_facts/total_supporting_facts*100:.1f}%)")

        # Debug: Show sample mismatches
        if found_supporting_facts < total_supporting_facts:
            print("\nDEBUG: Sample supporting facts NOT found:")
            count = 0
            for example in examples[:5]:
                for title, sent_id in example.supporting_facts:
                    key = (title, sent_id)
                    if key not in title_sent_to_para:
                        print(f"  Missing: title='{title}', sent_id={sent_id}")
                        count += 1
                        if count >= 5:
                            break
                if count >= 5:
                    break

            print("\nDEBUG: Sample paragraphs in corpus:")
            for i, para in enumerate(paragraphs[:5]):
                print(f"  Para {i}: title='{para.title}', sent_id={para.sentence_id}, id={para.para_id}")

        return ground_truth
