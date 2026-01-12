import re
import os
import urllib.request

# Base paths (relative to src/training/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def download_file(url, target_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def clean_liber_liber_text(text):
    """Aggressive cleaning for Liber Liber and general Italian texts."""
    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Precise markers for cleaning
    start_markers = [
        r"\*\*\* START OF THIS PROJECT GUTENBERG",
        r"\*\*\* START OF THE PROJECT GUTENBERG",
        r"\*\*\*START OF THIS PROJECT GUTENBERG",
        r"\*\*\* INIZIO DI QUESTO E-BOOK",
        r"\*\*\*INIZIO DI QUESTO E-BOOK",
        r"Digitized by the Internet Archive",
        r"\n-{5,}\n",  # Common Liber Liber header separator
    ]

    end_markers = [
        r"\*\*\* END OF THIS PROJECT GUTENBERG",
        r"\*\*\* END OF THE PROJECT GUTENBERG",
        r"\*\*\*END OF THIS PROJECT GUTENBERG",
        r"\*\*\* FINE DI QUESTO E-BOOK",
        r"\*\*\*FINE DI QUESTO E-BOOK",
        r"End of Project Gutenberg",
        r"End of the Project Gutenberg",
    ]

    # Remove header
    found_start = False
    for marker in start_markers:
        parts = re.split(marker, text, 1, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Find the first newline after the marker to avoid keeping the marker line
            text = parts[1].split("\n", 1)[-1]
            found_start = True
            break

    # Remove footer
    if found_start:
        for marker in end_markers:
            parts = re.split(marker, text, 1, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[0]
                break

    # Remove extra newlines and spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_directory(directory_path, output_file):
    all_text = []
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} not found.")
        return ""

    print(f"--- Processing Directory: {directory_path} ---")
    files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    for filename in sorted(files):
        filepath = os.path.join(directory_path, filename)
        print(f"Processing {filepath}...")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        if len(text.strip()) == 0:
            print(f"  -> Skipping empty file")
            continue

        text = clean_liber_liber_text(text)
        print(f"  -> {len(text):,} characters")
        all_text.append(text)

    combined = "\n\n\n".join(all_text)

    if not os.path.exists(DATA_CLEAN_DIR):
        os.makedirs(DATA_CLEAN_DIR)

    full_output_path = os.path.join(DATA_CLEAN_DIR, output_file)
    with open(full_output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"Saved combined text to {full_output_path}")
    print(f"Total: {len(combined):,} characters\n")
    return combined


import numpy as np
import json


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def train_bpe(text, vocab_size, max_chars=1_000_000):
    # Train on a subset for speed
    text_sample = text[:max_chars]
    num_merges = vocab_size - 256
    tokens = list(text_sample.encode("utf-8"))

    ids = list(tokens)
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        if (i + 1) % 50 == 0:
            print(f"Merge {i+1}/{num_merges} complete...")

    return merges


def encode_bpe(text, merges):
    tokens = list(text.encode("utf-8"))
    # Sort merges by their value (the order they were learned)
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    for pair, idx in sorted_merges:
        tokens = merge(tokens, pair, idx)
    return tokens


def main():
    VOCAB_SIZE = 512  # Smaller vocab for faster inference and smaller model

    # Ensure model output directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Process Directories
    pretrain_dir = os.path.join(DATA_RAW_DIR, "pretraining")
    pretrain_text = process_directory(pretrain_dir, "pretrain.txt")

    finetune_dir = os.path.join(DATA_RAW_DIR, "finetuning")
    finetune_text = process_directory(finetune_dir, "finetune.txt")

    print("--- Training BPE Tokenizer ---")
    combined_text = pretrain_text + "\n\n" + finetune_text
    merges = train_bpe(combined_text, VOCAB_SIZE)

    # Save BPE meta for JS
    json_merges = {f"{p[0]},{p[1]}": v for p, v in merges.items()}
    meta = {"merges": json_merges, "vocab_size": VOCAB_SIZE, "block_size": 256}
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Saved {meta_path} with BPE merges")

    # Tokenize and save as .bin
    print("\n--- Tokenizing Datasets ---")
    for name, text in [("pretrain", pretrain_text), ("finetune", finetune_text)]:
        ids = encode_bpe(text, merges)
        ids_np = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(DATA_CLEAN_DIR, f"{name}.bin")
        ids_np.tofile(out_path)
        print(f"Saved {len(ids):,} tokens to {out_path}")


if __name__ == "__main__":
    main()
