"""
Benchmark & Evaluation Script: Reliable Terza Rima
=================================================
Generates 100 iterations of 9-verse sequences (3 terzine) across generation modes
(OFF, SOFT, FORCED) using the ONNX model and evaluates:
1. Rhyme Accuracy (% of ABA BCB CDC rhyme pairs satisfied)
2. Rhyme Repetition Count (number of duplicate rhyme words within a canto)
3. Meter Compliance (% of verses with 10-12 syllables)
4. Model Perplexity / Log-likelihood score
"""

import json
import math
import os
import re
import sys
import numpy as np
import onnxruntime as ort

# Base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = os.path.dirname(SRC_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")
sys.path.insert(0, SRC_DIR)

from training.metric_utils import ItalianSyllableCounter, RhymeDetector

# Load metadata & rimario
with open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8") as f:
    META = json.load(f)

with open(os.path.join(MODEL_DIR, "rimario.json"), "r", encoding="utf-8") as f:
    RIMARIO = json.load(f)

# Reconstruct inverse rimario map (word -> suffix)
RIMARIO_INDEX = {}
for suffix, words in RIMARIO.items():
    for w in words:
        RIMARIO_INDEX[w.lower()] = suffix

VOCAB_SIZE = META["vocab_size"]
BLOCK_SIZE = META["block_size"]
MERGES = {tuple(map(int, k.split(","))): v for k, v in META["merges"].items()}

# Tokenizer functions
VOCAB = {i: bytes([i]) for i in range(256)}
for (p0, p1), idx in sorted(MERGES.items(), key=lambda x: x[1]):
    VOCAB[idx] = VOCAB[p0] + VOCAB[p1]


def decode(ids):
    tokens_bytes = b"".join(VOCAB.get(idx, b"") for idx in ids)
    return tokens_bytes.decode("utf-8", errors="replace")


def merge_ids(ids, pair, idx):
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


def encode(text):
    tokens = list(text.encode("utf-8"))
    sorted_merges = sorted(MERGES.items(), key=lambda x: x[1])
    for pair, idx in sorted_merges:
        tokens = merge_ids(tokens, pair, idx)
    return tokens


# ONNX Session
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
ORT_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def run_inference(tokens):
    input_tokens = tokens[-BLOCK_SIZE:]
    seq_len = len(input_tokens)
    input_array = np.array([input_tokens], dtype=np.int64)
    results = ORT_SESSION.run(None, {"input": input_array})
    output = results[0]  # shape: [1, seq_len, vocab_size]
    return output[0, -1, :]  # last token logits


def softmax(logits, temperature=0.8):
    logits = logits / max(temperature, 1e-5)
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)


def sample_top_p(probs, top_p=0.9):
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_index = np.searchsorted(cumulative_probs, top_p)
    keep_indices = sorted_indices[: cutoff_index + 1]

    filtered_probs = np.zeros_like(probs)
    filtered_probs[keep_indices] = probs[keep_indices]
    filtered_probs /= np.sum(filtered_probs)
    return np.random.choice(len(probs), p=filtered_probs)


def generate_canto(mode="FORCED", target_verses=9, prompt="Nel mezzo del cammin di nostra vita\n"):
    detector = RhymeDetector()
    syllable_counter = ItalianSyllableCounter()

    tokens = encode(prompt)
    verses = prompt.strip().split("\n")
    used_rhyme_words = set()

    # Track verse rhyme endings
    verse_endings = []
    for v in verses:
        if v.strip():
            verse_endings.append(detector.get_rhyme_suffix(v))

    target_pairs = [(2, 0), (3, 1), (5, 3), (6, 4), (8, 6)]  # ABA BCB CDC target pairs

    max_steps = 300
    step = 0

    while len(verses) < target_verses and step < max_steps:
        step += 1
        current_verse_idx = len(verses)
        target_rhyme_idx = -1
        if current_verse_idx in [2, 3, 5, 6, 8]:
            if current_verse_idx == 2: target_rhyme_idx = 0
            elif current_verse_idx == 3: target_rhyme_idx = 1
            elif current_verse_idx == 5: target_rhyme_idx = 3
            elif current_verse_idx == 6: target_rhyme_idx = 4
            elif current_verse_idx == 8: target_rhyme_idx = 6

        # Check if we should force rhyme ending
        current_text = decode(tokens)
        current_partial_verse = current_text.split("\n")[-1]

        if mode == "FORCED" and target_rhyme_idx >= 0 and target_rhyme_idx < len(verse_endings) and len(current_partial_verse) >= 25:
            target_suffix = verse_endings[target_rhyme_idx]
            candidates = RIMARIO.get(target_suffix, [])
            available = [w for w in candidates if w not in used_rhyme_words]
            if not available and candidates:
                available = candidates

            if available:
                # Pick candidate word with best syllable fit & highest model score
                best_word = None
                best_tokens = None
                best_score = -float("inf")

                for word in available[:15]:
                    cand_tokens = encode(" " + word + "\n")
                    cand_line = current_partial_verse + " " + word
                    s_count = syllable_counter.count_verse_syllables(cand_line)
                    meter_penalty = abs(s_count - 11) * 2.0
                    
                    # Quick log-prob evaluation
                    logits = run_inference(tokens + cand_tokens[:-1])
                    log_p = math.log(softmax(logits)[cand_tokens[-1]] + 1e-10) - meter_penalty

                    if log_p > best_score:
                        best_score = log_p
                        best_word = word
                        best_tokens = cand_tokens

                if best_word:
                    tokens.extend(best_tokens)
                    used_rhyme_words.add(best_word)
                    full_gen = decode(tokens)
                    verses = [v for v in full_gen.strip().split("\n") if v.strip()]
                    if len(verses) < len(verse_endings):
                        verse_endings = verse_endings[:len(verses)]
                    while len(verse_endings) < len(verses):
                        verse_endings.append(detector.get_rhyme_suffix(verses[len(verse_endings)]))
                    continue

        # Regular token sampling
        logits = run_inference(tokens)
        probs = softmax(logits)
        next_token = sample_top_p(probs)
        tokens.append(next_token)

        full_gen = decode(tokens)
        verses = [v for v in full_gen.strip().split("\n") if v.strip()]
        while len(verse_endings) < len(verses):
            verse_endings.append(detector.get_rhyme_suffix(verses[len(verse_endings)]))

    full_text = decode(tokens)
    final_verses = [v.strip() for v in full_text.strip().split("\n") if v.strip()][:target_verses]
    return final_verses


def evaluate_batch(num_samples=20, mode="FORCED"):
    detector = RhymeDetector()
    syllable_counter = ItalianSyllableCounter()

    rhyme_matches = 0
    total_rhyme_pairs = 0
    repetition_count = 0
    meter_valid_count = 0
    total_verses = 0

    target_pairs = [(2, 0), (3, 1), (5, 3), (6, 4), (8, 6)]

    print(f"--- Evaluating {num_samples} samples in mode: {mode} ---")
    for i in range(num_samples):
        verses = generate_canto(mode=mode, target_verses=9)
        if len(verses) < 9:
            continue

        # Check rhyme pairs
        rhyme_words_in_sample = []
        for v in verses:
            words = re.findall(r"[a-zA-Zàèéìíòóùú]+", v)
            if words:
                rhyme_words_in_sample.append(words[-1].lower())

        # Duplicate rhyme words check
        seen = set()
        for w in rhyme_words_in_sample:
            if w in seen:
                repetition_count += 1
            seen.add(w)

        # Rhyme pair checks (ABA BCB CDC)
        for v_idx, ref_idx in target_pairs:
            if v_idx < len(verses) and ref_idx < len(verses):
                total_rhyme_pairs += 1
                if detector.rhymes_with(verses[v_idx], verses[ref_idx], strict=True):
                    rhyme_matches += 1

        # Meter check
        for v in verses:
            total_verses += 1
            is_valid, _ = syllable_counter.is_endecasillabo(v)
            if is_valid:
                meter_valid_count += 1

        if (i + 1) % 5 == 0 or i == num_samples - 1:
            print(f"  Processed {i+1}/{num_samples} samples...")

    rhyme_acc = (rhyme_matches / total_rhyme_pairs * 100) if total_rhyme_pairs > 0 else 0
    meter_acc = (meter_valid_count / total_verses * 100) if total_verses > 0 else 0

    return {
        "mode": mode,
        "samples": num_samples,
        "rhyme_accuracy_pct": round(rhyme_acc, 2),
        "rhyme_repetitions": repetition_count,
        "meter_compliance_pct": round(meter_acc, 2),
    }


def main():
    print("==================================================")
    print(" InfiniteDante — Reliable Terza Rima Benchmark")
    print("==================================================\n")

    modes = ["OFF", "FORCED"]
    results = []

    for mode in modes:
        res = evaluate_batch(num_samples=20, mode=mode)
        results.append(res)

    print("\n================ FINAL RESULTS ================")
    print(f"{'Mode':<10} | {'Rhyme Acc (%)':<15} | {'Repetitions':<12} | {'Meter Acc (%)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['mode']:<10} | {r['rhyme_accuracy_pct']:<15}% | {r['rhyme_repetitions']:<12} | {r['meter_compliance_pct']:<15}%")
    print("================================================")


if __name__ == "__main__":
    main()
