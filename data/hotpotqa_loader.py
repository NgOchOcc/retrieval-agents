"""
HotpotQA dataset loader.
Handles data loading, corpus construction, and gold supporting facts extraction.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class HotpotQAExample:
    """Single HotpotQA example."""
    question: str
    gold_titles: List[str]
    example_id: str


@dataclass
class Passage:
    """Wikipedia passage with title and text."""
    title: str
    text: str
    doc_id: int


class HotpotQALoader:
    """Loads and processes HotpotQA dataset."""

    def __init__(
        self,
        split: str = "validation",
        dataset_type: str = "fullwiki",
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            split: Dataset split ('train' or 'validation')
            dataset_type: 'fullwiki' or 'distractor'
            max_samples: Limit number of samples (for debugging)
            seed: Random seed for subsampling
        """
        self.split = split
        self.dataset_type = dataset_type
        self.max_samples = max_samples
        self.seed = seed

    def load_examples(self) -> List[HotpotQAExample]:
        """
        Load HotpotQA examples with questions and gold supporting facts.

        Returns:
            List of HotpotQAExample objects
        """
        print(f"Loading HotpotQA {self.dataset_type} - {self.split} split...")

        # Load dataset
        dataset = load_dataset("hotpot_qa", self.dataset_type, split=self.split)

        # Subsample if requested
        if self.max_samples is not None and self.max_samples < len(dataset):
            dataset = dataset.shuffle(seed=self.seed).select(range(self.max_samples))
            print(f"Subsampled to {self.max_samples} examples")

        examples = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing examples")):
            # Extract gold supporting fact titles
            gold_titles = self._extract_gold_titles(item['supporting_facts'])

            examples.append(HotpotQAExample(
                question=item['question'],
                gold_titles=gold_titles,
                example_id=item['id']
            ))

        print(f"Loaded {len(examples)} examples")
        return examples

    def build_corpus(self) -> List[Passage]:
        """
        Build corpus of Wikipedia passages from HotpotQA.

        Returns:
            List of Passage objects with title and text
        """
        print(f"Building corpus from HotpotQA {self.dataset_type}...")

        dataset = load_dataset("hotpot_qa", self.dataset_type, split=self.split)

        # Use dict to deduplicate passages by (title, text)
        corpus_dict: Dict[Tuple[str, str], None] = {}

        for item in tqdm(dataset, desc="Extracting passages"):
            context = item['context']

            # Each context has multiple [title, sentences] pairs
            for title, sentences in context['sentences']:
                # Join sentences into single passage
                text = " ".join(sentences)

                if text.strip():  # Only add non-empty passages
                    corpus_dict[(title, text)] = None

        # Convert to list of Passage objects
        passages = [
            Passage(title=title, text=text, doc_id=idx)
            for idx, (title, text) in enumerate(corpus_dict.keys())
        ]

        print(f"Built corpus with {len(passages)} unique passages")
        return passages

    def _extract_gold_titles(self, supporting_facts: Dict) -> List[str]:
        """
        Extract unique gold document titles from supporting facts.

        Args:
            supporting_facts: Dict with 'title' and 'sent_id' lists

        Returns:
            Deduplicated list of gold titles
        """
        titles = supporting_facts['title']
        # Deduplicate while preserving order
        seen = set()
        unique_titles = []
        for title in titles:
            if title not in seen:
                seen.add(title)
                unique_titles.append(title)
        return unique_titles
