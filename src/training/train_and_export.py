import os
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

# Base paths (relative to src/training/)
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
PRETRAIN_LR = 3e-4 # Slightly lower for deeper model
FINETUNE_ITERS = 2000  # Reduced to prevent overfitting
FINETUNE_LR = 5e-5     # Much lower LR for fine-tuning

EVAL_INTERVAL = 500
EARLY_STOPPING_PATIENCE_PRETRAIN = 5   # Stricter for pretraining
EARLY_STOPPING_PATIENCE_FINETUNE = 10  # More lenient for finetuning
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# ==============================================================================
# Tokenizer & Data Loading
# ==============================================================================
# Load metadata from prepare_data.py
meta_path = os.path.join(MODEL_DIR, 'meta.json')
if not os.path.exists(meta_path):
    print(f"Error: {meta_path} not found. Run prepare_data.py first.")
    import sys
    sys.exit(1)

with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
vocab_size = meta['vocab_size']
merges_json = meta['merges']

# Reconstruct merges and vocab for decoding
merges = {tuple(map(int, k.split(','))): v for k, v in merges_json.items()}
vocab = {i: bytes([i]) for i in range(256)}
for (p0, p1), idx in sorted(merges.items(), key=lambda x: x[1]):
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens_bytes = b"".join(vocab[idx] for idx in ids)
    return tokens_bytes.decode("utf-8", errors="replace")

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
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
        tokens = merge(tokens, pair, idx)
    return tokens

# Load .bin files
def load_bin(filename):
    path = os.path.join(DATA_CLEAN_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))

print("Loading tokenized datasets...")
pretrain_data_raw = load_bin('pretrain.bin')
finetune_data_raw = load_bin('finetune.bin')

def get_split_tensors(data):
    n = int(0.9 * len(data))
    return data[:n], data[n:]

# ==============================================================================
# DataLoader
# ==============================================================================
def get_batch(data_split):
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
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
# Model Architecture (NanoGPT)
# ==============================================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(DROPOUT),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# Training Loop
# ==============================================================================
model = NanoGPT(vocab_size).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def train_stage(stage_name, train_data, val_data, iters, lr, patience):
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

# Stage 1: Pre-training
train_pre, val_pre = get_split_tensors(pretrain_data_raw)
pretrain_train_losses, pretrain_val_losses = train_stage(
    "Pre-training (General Italian)", train_pre, val_pre, 
    PRETRAIN_ITERS, PRETRAIN_LR, EARLY_STOPPING_PATIENCE_PRETRAIN
)

# Stage 2: Fine-tuning
train_fine, val_fine = get_split_tensors(finetune_data_raw)
finetune_train_losses, finetune_val_losses = train_stage(
    "Fine-tuning (Dante)", train_fine, val_fine, 
    FINETUNE_ITERS, FINETUNE_LR, EARLY_STOPPING_PATIENCE_FINETUNE
)

# ==============================================================================
# Save Loss Plots
# ==============================================================================
print("\n--- Saving Loss Plots ---")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pre-training losses
ax1.plot(pretrain_train_losses, label='Train Loss', marker='o', markersize=3)
ax1.plot(pretrain_val_losses, label='Val Loss', marker='s', markersize=3)
ax1.set_xlabel('Evaluation Step')
ax1.set_ylabel('Loss')
ax1.set_title('Pre-training Loss (General Italian)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Fine-tuning losses
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

# Also save combined loss history as JSON for later analysis
loss_history = {
    'pretrain': {'train': pretrain_train_losses, 'val': pretrain_val_losses},
    'finetune': {'train': finetune_train_losses, 'val': finetune_val_losses}
}
loss_history_path = os.path.join(MODEL_DIR, 'loss_history.json')
with open(loss_history_path, 'w') as f:
    json.dump(loss_history, f, indent=2)
print(f"Saved {loss_history_path}")

# Final Generation Sample
print("\n--- Final Sample Generation ---")
context_str = "Nel mezzo del cammin di nostra vita "
context = torch.tensor([encode(context_str)], dtype=torch.long, device=DEVICE)
generated = model.generate(context, max_new_tokens=100)
print(decode(generated[0].tolist()))

# Export to ONNX
print("\nExporting to ONNX...")
class ONNXWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, idx): return self.m(idx)[0]

onnx_model = ONNXWrapper(model).eval()
dummy_input = torch.randint(0, vocab_size, (1, BLOCK_SIZE), dtype=torch.long).to(DEVICE)
onnx_path = os.path.join(MODEL_DIR, 'model.onnx')
torch.onnx.export(onnx_model, dummy_input, onnx_path, export_params=True, opset_version=14, 
                  do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'}, 'output': {0: 'batch_size', 1: 'sequence'}})
print(f"Saved {onnx_path}")
