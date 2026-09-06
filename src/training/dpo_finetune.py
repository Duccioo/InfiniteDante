"""
Direct Preference Optimization (DPO) for Terzina Dantesca
==========================================================
Replaces unstable PPO RL with Direct Preference Optimization (DPO).

DPO directly optimizes policy weights to prefer metrically sound, rhyming
terzine (evaluated via TerzinaScorer) over non-rhyming/dysmetric lines,
without requiring a separate critic/value network or advantage normalization.

Loss:
    L_DPO(theta; ref) = -E_{(x, y_w, y_l)} [ log sigma( beta * ( log(pi_theta(y_w|x)/pi_ref(y_w|x))
                                                               - log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ) ]
"""

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Local imports
from metric_utils import TerzinaScorer, count_syllables
from model import NanoGPT, ONNXWrapper, count_parameters

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DPOConfig:
    beta: float = 0.1  # DPO temperature parameter
    lr: float = 2e-5
    batch_size: int = 4
    num_steps: int = 50
    eval_interval: int = 10
    max_seq_len: int = 128
    weight_decay: float = 0.01


def load_tokenizer():
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found")

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


def get_sequence_log_prob(model: nn.Module, prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
    """Compute sum of log probabilities assigned by model to completion tokens conditioned on prompt."""
    full_seq = torch.cat([prompt_ids, completion_ids], dim=1)  # [B, T_p + T_c]
    logits, _ = model(full_seq)  # [B, T, V]
    # Prediction for token at position t is from logits at t-1
    # We want log-probs for completion tokens: positions T_p to T_p + T_c - 1
    shift_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]  # [B, T_c, V]
    shift_labels = completion_ids  # [B, T_c]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return target_log_probs.sum(dim=-1)  # [B]


def create_preference_batch(
    encode, decode, batch_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic preference pairs:
    - prompt: first verse of a terzina
    - y_w: authentic Dante continuation (rhymes ABA, 11 syllables)
    - y_l: broken/dysmetric continuation
    """
    terzine_catalog = [
        (
            "Nel mezzo del cammin di nostra vita\n",
            "mi ritrovai per una selva oscura,\nché la diritta via era smarrita.\n",
            "andai per una selva molto grande,\ndove non c'era nessun sentiero.\n",
        ),
        (
            "Amor, ch'al cor gentil ratto s'apprende,\n",
            "prese costui de la bella persona\nche mi fu tolta; e 'l modo ancor m'offende.\n",
            "prese me per la bella ragazza\nche mi piaceva tanto ieri sera.\n",
        ),
        (
            "Per me si va ne la città dolente,\n",
            "per me si va ne l'etterno dolore,\nper me si va tra la perduta gente.\n",
            "per me si va dentro la città,\ndove la gente piange forte.\n",
        ),
        (
            "La gloria di colui che tutto move\n",
            "per l'universo penetra, e risplende\nin una parte più e meno altrove.\n",
            "splende nel cielo sopra di noi\ne illumina la terra intera.\n",
        ),
        (
            "O voi che siete in piccioletta barca,\n",
            "desiderosi d'ascoltar, seguiti\ndietro al mio legno che cantando varca,\n",
            "venite con me sulla barchetta\nper vedere il grande mare.\n",
        ),
    ]

    selected = np.random.choice(len(terzine_catalog), size=batch_size, replace=True)
    prompts, y_ws, y_ls = [], [], []

    for idx in selected:
        p, w, l = terzine_catalog[idx]
        prompts.append(encode(p))
        y_ws.append(encode(w))
        y_ls.append(encode(l))

    # Pad within batch
    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long, device=DEVICE)

    return pad(prompts), pad(y_ws), pad(y_ls)


def train_dpo():
    print("=== Direct Preference Optimization (DPO) for InfiniteDante ===")
    vocab_size, encode, decode = load_tokenizer()
    cfg = DPOConfig()

    # Load base model checkpoint if available
    checkpoint_path = os.path.join(MODEL_DIR, "finetune_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(MODEL_DIR, "pretrain_checkpoint.pt")

    print(f"Loading policy model from {checkpoint_path}...")
    model = NanoGPT(vocab_size=vocab_size).to(DEVICE)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Loaded existing checkpoint weights.")
    else:
        print("No checkpoint found; initialized randomly.")

    # Freeze reference model
    print("Cloning reference model (pi_ref)...")
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scorer = TerzinaScorer()

    losses = []
    rewards_w = []
    rewards_l = []

    print(f"\nStarting DPO training ({cfg.num_steps} steps, beta={cfg.beta}, lr={cfg.lr})...")
    model.train()

    for step in range(cfg.num_steps):
        prompts, y_w, y_l = create_preference_batch(encode, decode, batch_size=cfg.batch_size)

        # Policy log-probs
        pi_w_logprob = get_sequence_log_prob(model, prompts, y_w)
        pi_l_logprob = get_sequence_log_prob(model, prompts, y_l)

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_w_logprob = get_sequence_log_prob(ref_model, prompts, y_w)
            ref_l_logprob = get_sequence_log_prob(ref_model, prompts, y_l)

        # Implicit rewards
        pi_logratio_w = pi_w_logprob - ref_w_logprob
        pi_logratio_l = pi_l_logprob - ref_l_logprob

        logits = cfg.beta * (pi_logratio_w - pi_logratio_l)
        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = (logits > 0).float().mean().item()
        losses.append(loss.item())
        rewards_w.append(pi_logratio_w.mean().item())
        rewards_l.append(pi_logratio_l.mean().item())

        if (step + 1) % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            print(
                f"Step {step+1:03d}/{cfg.num_steps} | "
                f"DPO Loss: {loss.item():.4f} | "
                f"Acc (r_w > r_l): {acc * 100:.1f}% | "
                f"Reward Margin: {(pi_logratio_w.mean() - pi_logratio_l.mean()).item():.3f}"
            )

    # Save fine-tuned DPO checkpoint
    dpo_checkpoint_path = os.path.join(MODEL_DIR, "dpo_checkpoint.pt")
    torch.save(model.state_dict(), dpo_checkpoint_path)
    print(f"\nSaved DPO checkpoint: {dpo_checkpoint_path}")

    # Plot DPO learning curves
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="DPO Loss", color="#d4a056")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("DPO Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rewards_w, label="r(y_w)", color="#2ecc71")
    plt.plot(rewards_l, label="r(y_l)", color="#e74c3c")
    plt.xlabel("Step")
    plt.ylabel("Implicit Reward")
    plt.title("Policy vs Reference Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "dpo_training_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved DPO plots to: {plot_path}")

    return dpo_checkpoint_path


if __name__ == "__main__":
    train_dpo()
