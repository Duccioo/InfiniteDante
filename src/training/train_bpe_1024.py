"""
Train 1024-Vocab BPE Tokenizer for InfiniteDante
================================================
Trains a byte-level BPE tokenizer with 1024 vocabulary size on the
authentic Dante poetic corpus.
Outputs:
- model/meta_1024.json
- data/clean/poetic_1024.bin
"""

import json
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "clean", "trecento_poetic_corpus.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_META = os.path.join(MODEL_DIR, "meta_1024.json")
OUTPUT_BIN = os.path.join(DATA_CLEAN_DIR, "poetic_1024.bin")

VOCAB_SIZE = 1024
BLOCK_SIZE = 512


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    p0, p1 = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def train_bpe():
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: {CORPUS_PATH} not found. Run prepare_poetic_corpus.py first.")
        sys.exit(1)

    print(f"Loading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Corpus size: {len(text):,} characters")
    num_merges = VOCAB_SIZE - 256
    print(f"Target vocab size: {VOCAB_SIZE} ({num_merges} merges)")

    # Sample text for fast BPE merge learning if very large
    sample_text = text[:800_000]
    ids = list(sample_text.encode("utf-8"))

    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, best_pair, idx)
        merges[best_pair] = idx

        if (i + 1) % 100 == 0 or (i + 1) == num_merges:
            # Decode the newly learned token representation
            try:
                p0_b = bytes([best_pair[0]]) if best_pair[0] < 256 else b".."
                p1_b = bytes([best_pair[1]]) if best_pair[1] < 256 else b".."
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {idx}")
            except Exception:
                print(f"Merge {i+1}/{num_merges} complete...")

    # Save meta_1024.json
    json_merges = {f"{p[0]},{p[1]}": v for p, v in merges.items()}
    meta = {
        "merges": json_merges,
        "vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved tokenizer metadata: {OUTPUT_META}")

    # Encode full dataset
    print("\nTokenizing full corpus...")
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    full_ids = list(text.encode("utf-8"))
    for pair, idx in sorted_merges:
        full_ids = merge(full_ids, pair, idx)

    print(f"Original byte count: {len(text.encode('utf-8')):,}")
    print(f"Tokenized token count: {len(full_ids):,}")
    print(f"Compression ratio: {len(text.encode('utf-8')) / len(full_ids):.2f}x")

    ids_np = np.array(full_ids, dtype=np.uint16)
    ids_np.tofile(OUTPUT_BIN)
    print(f"Saved tokenized binary dataset: {OUTPUT_BIN}")


if __name__ == "__main__":
    train_bpe()
