"""
Two-Stage Training Pipeline for InfiniteDante
==============================================
Stage 1: Pre-training on general Italian text
Stage 2: Fine-tuning on Dante's works

Checkpoints are saved after each stage for RL fine-tuning.

Usage:
    python train_and_export.py                    # Full training
    python train_and_export.py --skip-pretrain    # Skip pre-training (use checkpoint)
"""

import os
import sys
import json
import argparse
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from model import NanoGPT, ONNXWrapper, count_parameters

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'clean')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ==============================================================================
# Hyperparameters (Optimized for ~8.5M params)
# ==============================================================================
BATCH_SIZE = 32
BLOCK_SIZE = 256
N_EMBD = 256
N_HEAD = 8
N_LAYER = 10
DROPOUT = 0.2

# Two-stage Training Config
PRETRAIN_ITERS = 10000 
PRETRAIN_LR = 3e-4
FINETUNE_ITERS = 2000
FINETUNE_LR = 5e-5

EVAL_INTERVAL = 500
EARLY_STOPPING_PATIENCE_PRETRAIN = 5
EARLY_STOPPING_PATIENCE_FINETUNE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# ==============================================================================
# Tokenizer & Data Loading
# ==============================================================================
def load_tokenizer():
    """Load BPE tokenizer from meta.json."""
    meta_path = os.path.join(MODEL_DIR, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run prepare_data.py first.")
        sys.exit(1)
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    vocab_size = meta['vocab_size']
    merges_json = meta['merges']
    
    # Reconstruct merges and vocab
    merges = {tuple(map(int, k.split(','))): v for k, v in merges_json.items()}
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in sorted(merges.items(), key=lambda x: x[1]):
        vocab[idx] = vocab[p0] + vocab[p1]
    
    def decode(ids):
        tokens_bytes = b"".join(vocab.get(idx, b'') for idx in ids)
        return tokens_bytes.decode("utf-8", errors="replace")
    
    def merge_ids(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
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


def load_bin(filename):
    """Load tokenized data from .bin file."""
    path = os.path.join(DATA_CLEAN_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))


def get_split_tensors(data):
    """Split data into train/val (90/10)."""
    n = int(0.9 * len(data))
    return data[:n], data[n:]


# ==============================================================================
# DataLoader
# ==============================================================================
def get_batch(data_split):
    """Sample a batch of sequences."""
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    """Estimate train and validation loss."""
    out = {}
    model.eval()
    for split_name, data_split in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out


# ==============================================================================
# Training Loop
# ==============================================================================
def train_stage(model, stage_name, train_data, val_data, iters, lr, patience):
    """Train with early stopping and loss tracking."""
    print(f"\n>>> Stage: {stage_name} ({iters} iters, lr={lr}, patience={patience})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for iter in range(iters):
        if iter % EVAL_INTERVAL == 0 or iter == iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
            
            print(f"Step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Early stopping check
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"  -> Early stopping triggered at step {iter}!")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print(f"  -> Restored best model with val loss {best_val_loss:.4f}")
                    break
        
        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return train_losses, val_losses


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Two-stage training for InfiniteDante')
    parser.add_argument('--skip-pretrain', action='store_true', 
                        help='Skip pre-training, load from checkpoint')
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    vocab_size, encode, decode = load_tokenizer()
    print(f"Vocab size: {vocab_size}")
    
    # Load data
    print("\nLoading tokenized datasets...")
    pretrain_data_raw = load_bin('pretrain.bin')
    finetune_data_raw = load_bin('finetune.bin')
    
    if pretrain_data_raw is None or finetune_data_raw is None:
        print("Error: Data files not found. Run prepare_data.py first.")
        sys.exit(1)
    
    # Create model
    print("\nCreating model...")
    model = NanoGPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Checkpoint paths
    pretrain_checkpoint = os.path.join(MODEL_DIR, 'pretrain_checkpoint.pt')
    finetune_checkpoint = os.path.join(MODEL_DIR, 'finetune_checkpoint.pt')
    
    # Stage 1: Pre-training
    if args.skip_pretrain and os.path.exists(pretrain_checkpoint):
        print(f"\n>>> Skipping pre-training, loading checkpoint: {pretrain_checkpoint}")
        model.load_state_dict(torch.load(pretrain_checkpoint, map_location=DEVICE))
        pretrain_train_losses, pretrain_val_losses = [], []
    else:
        train_pre, val_pre = get_split_tensors(pretrain_data_raw)
        pretrain_train_losses, pretrain_val_losses = train_stage(
            model, "Pre-training (General Italian)", train_pre, val_pre, 
            PRETRAIN_ITERS, PRETRAIN_LR, EARLY_STOPPING_PATIENCE_PRETRAIN
        )
        
        # Save pre-training checkpoint
        torch.save(model.state_dict(), pretrain_checkpoint)
        print(f"\n>>> Saved pre-training checkpoint: {pretrain_checkpoint}")
    
    # Stage 2: Fine-tuning
    train_fine, val_fine = get_split_tensors(finetune_data_raw)
    finetune_train_losses, finetune_val_losses = train_stage(
        model, "Fine-tuning (Dante)", train_fine, val_fine, 
        FINETUNE_ITERS, FINETUNE_LR, EARLY_STOPPING_PATIENCE_FINETUNE
    )
    
    # Save fine-tuning checkpoint
    torch.save(model.state_dict(), finetune_checkpoint)
    print(f"\n>>> Saved fine-tuning checkpoint: {finetune_checkpoint}")
    
    # ==============================================================================
    # Save Loss Plots
    # ==============================================================================
    print("\n--- Saving Loss Plots ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if pretrain_train_losses:
        ax1.plot(pretrain_train_losses, label='Train Loss', marker='o', markersize=3)
        ax1.plot(pretrain_val_losses, label='Val Loss', marker='s', markersize=3)
        ax1.set_xlabel('Evaluation Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Pre-training Loss (General Italian)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Pre-training skipped\n(loaded checkpoint)', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Pre-training Loss (Skipped)')
    
    ax2.plot(finetune_train_losses, label='Train Loss', marker='o', markersize=3)
    ax2.plot(finetune_val_losses, label='Val Loss', marker='s', markersize=3)
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Fine-tuning Loss (Dante)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(MODEL_DIR, 'loss_plots.png')
    plt.savefig(loss_plot_path, dpi=150)
    print(f"Saved {loss_plot_path}")
    
    # Save loss history
    loss_history = {
        'pretrain': {'train': pretrain_train_losses, 'val': pretrain_val_losses},
        'finetune': {'train': finetune_train_losses, 'val': finetune_val_losses}
    }
    loss_history_path = os.path.join(MODEL_DIR, 'loss_history.json')
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Saved {loss_history_path}")
    
    # ==============================================================================
    # Final Generation Sample
    # ==============================================================================
    print("\n--- Final Sample Generation ---")
    model.eval()
    context_str = "Nel mezzo del cammin di nostra vita "
    context = torch.tensor([encode(context_str)], dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=100)
    print(decode(generated[0].tolist()))
    
    # ==============================================================================
    # Export to ONNX
    # ==============================================================================
    print("\nExporting to ONNX...")
    onnx_wrapper = ONNXWrapper(model).eval()
    dummy_input = torch.randint(0, vocab_size, (1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
    onnx_path = os.path.join(MODEL_DIR, 'model.onnx')
    
    torch.onnx.export(
        onnx_wrapper, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )
    print(f"Saved {onnx_path}")
    
    print("\n=== Training Complete ===")
    print(f"Checkpoints saved:")
    print(f"  - Pre-train: {pretrain_checkpoint}")
    print(f"  - Fine-tune: {finetune_checkpoint}")
    print(f"\nTo run RL fine-tuning for metric enforcement:")
    print(f"  python rl_finetune.py")


if __name__ == '__main__':
    main()
