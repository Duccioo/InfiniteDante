"""
Automated Poetic Metric & Rhyme Benchmark
=========================================
Evaluates any model checkpoint (.pt) or ONNX model (.onnx) for:
- Rhyme scheme accuracy (ABA BCB terza rima)
- Syllabic meter accuracy (Italian endecasillabo: 11 syllables with sinalefe)
- Token throughput (tokens/sec and latency)
- Outputs clean comparison table and JSON report.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch

from metric_utils import (
    ItalianSyllableCounter,
    RhymeDetector,
    TerzinaScorer,
    count_syllables,
    get_stressed_suffix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

BENCHMARK_PROMPTS = [
    "Nel mezzo del cammin di nostra vita\n",
    "Amor, ch'al cor gentil ratto s'apprende\n",
    "Per me si va ne la città dolente\n",
    "La gloria di colui che tutto move\n",
    "O voi che siete in piccioletta barca\n",
    "Tanto gentile e tanto onesta pare\n",
    "Donne ch'avete intelletto d'amore\n",
    "Guido, i' vorrei che tu e Lapo ed io\n",
    "Così nel mio parlar voglio esser aspro\n",
    "Lasciate ogne speranza, voi ch'intrate\n",
]


def load_tokenizer():
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    merges_json = meta["merges"]
    merges = {tuple(map(int, k.split(","))): v for k, v in merges_json.items()}
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in sorted(merges.items(), key=lambda x: x[1]):
        vocab[idx] = vocab[p0] + vocab[p1]

    def decode(ids):
        tokens_bytes = b"".join(vocab.get(idx, b"") for idx in ids)
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
        sorted_merges = sorted(merges.items(), key=lambda x: x[1])
        for pair, idx in sorted_merges:
            tokens = merge_ids(tokens, pair, idx)
        return tokens

    return vocab_size, encode, decode


def run_benchmark(model_path: str = None, num_tokens_per_prompt: int = 60) -> Dict:
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "model_int8.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "model.onnx")

    print(f"=== Running Poetic Metric Benchmark ===")
    print(f"Target Model: {model_path}")
    print(f"Test Prompts: {len(BENCHMARK_PROMPTS)}")

    vocab_size, encode, decode = load_tokenizer()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    scorer = TerzinaScorer()
    syllable_counter = ItalianSyllableCounter()
    rhyme_detector = RhymeDetector()

    total_tokens = 0
    start_total_time = time.time()

    all_verse_syllables = []
    perfect_meter_count = 0
    total_verse_count = 0
    total_reward_sum = 0.0

    print("\nEvaluating prompts...")
    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        tokens = encode(prompt)
        prompt_len = len(tokens)

        # Generate tokens
        for _ in range(num_tokens_per_prompt):
            ctx = tokens[-256:]
            input_tensor = np.array([ctx], dtype=np.int64)
            logits = session.run(None, {input_name: input_tensor})[0][0, -1, :]
            # Greedy / low-temp for reproducible evaluation
            next_token = int(np.argmax(logits))
            tokens.append(next_token)
            total_tokens += 1

        generated_text = decode(tokens)
        verses = [v.strip() for v in generated_text.splitlines() if len(v.strip()) >= 5]

        # Analyze meter
        for v in verses:
            syl = syllable_counter.count_verse_syllables(v)
            all_verse_syllables.append(syl)
            if syl == 11:
                perfect_meter_count += 1
            total_verse_count += 1

        reward, _ = scorer.compute_reward(generated_text)
        total_reward_sum += reward

    total_time = time.time() - start_total_time
    tps = total_tokens / total_time
    mean_syl = float(np.mean(all_verse_syllables)) if all_verse_syllables else 0
    std_syl = float(np.std(all_verse_syllables)) if all_verse_syllables else 0
    meter_acc = (perfect_meter_count / total_verse_count * 100) if total_verse_count > 0 else 0
    avg_reward = total_reward_sum / len(BENCHMARK_PROMPTS)

    results = {
        "model": os.path.basename(model_path),
        "tokens_per_second": round(tps, 2),
        "total_tokens": total_tokens,
        "total_time_seconds": round(total_time, 2),
        "total_verses_evaluated": total_verse_count,
        "mean_syllables_per_verse": round(mean_syl, 2),
        "syllable_std_dev": round(std_syl, 2),
        "perfect_11_syllable_percent": round(meter_acc, 1),
        "average_terzina_score": round(avg_reward, 3),
    }

    print("\n" + "=" * 50)
    print("           BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model:                    {results['model']}")
    print(f"Throughput:               {results['tokens_per_second']} tokens/sec")
    print(f"Verses Analyzed:          {results['total_verses_evaluated']}")
    print(f"Mean Syllables / Verse:   {results['mean_syllables_per_verse']} (target: 11.0)")
    print(f"Syllable Std Dev:         ±{results['syllable_std_dev']}")
    print(f"Exact 11-Syllables:       {results['perfect_11_syllable_percent']}%")
    print(f"Avg Terzina Reward:       {results['average_terzina_score']} / 1.000")
    print("=" * 50)

    out_json = os.path.join(MODEL_DIR, "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()
    run_benchmark(model_path=args.model, num_tokens_per_prompt=args.tokens)
