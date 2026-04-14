"""Debug script to check evaluation issues."""

from data_loader import HotpotQADataLoader

# Load a small sample
data_loader = HotpotQADataLoader(split="test", config="fullwiki", cache_dir="./cache")

print("Loading examples...")
examples = data_loader.load_dataset(max_samples=10)

print(f"\nLoaded {len(examples)} examples")
print("\nSample example:")
ex = examples[0]
print(f"  Question ID: {ex.question_id}")
print(f"  Question: {ex.question}")
print(f"  Answer: {ex.answer}")
print(f"  Supporting facts: {ex.supporting_facts[:3]}")
print(f"  Context: {len(ex.context)} documents")

# Check corpus
print("\n" + "="*60)
print("Checking corpus...")

# Build corpus from examples (test mode)
paragraphs, supporting_map = data_loader.build_corpus_from_examples(examples)

print(f"\nCorpus statistics:")
print(f"  Total paragraphs: {len(paragraphs)}")
print(f"  Questions with supporting facts: {len(supporting_map)}")

# Check a specific question
if supporting_map:
    qid = list(supporting_map.keys())[0]
    print(f"\nSample question ID: {qid}")
    print(f"  Number of supporting paragraphs: {len(supporting_map[qid])}")
    print(f"  Supporting paragraph IDs: {list(supporting_map[qid])[:5]}")

# Check ground truth labels
print("\n" + "="*60)
print("Checking ground truth...")

ground_truth = data_loader.get_ground_truth_labels(examples, paragraphs)

print(f"\nGround truth statistics:")
print(f"  Total questions: {len(ground_truth)}")

if ground_truth:
    qid = list(ground_truth.keys())[0]
    print(f"\nSample question: {qid}")
    print(f"  Relevant docs: {len(ground_truth[qid])}")
    print(f"  Labels: {ground_truth[qid]}")

# Check paragraph IDs format
print("\n" + "="*60)
print("Checking paragraph ID format...")
print(f"\nSample paragraph IDs from corpus:")
for i, p in enumerate(paragraphs[:5]):
    print(f"  {i+1}. {p.para_id} - {p.title[:30]}")

print("\n" + "="*60)
print("Summary:")
print(f"  ✓ Examples loaded: {len(examples)}")
print(f"  ✓ Paragraphs extracted: {len(paragraphs)}")
print(f"  ✓ Questions with ground truth: {len(ground_truth)}")
print(f"  ✓ Supporting facts mapped: {len(supporting_map)}")
