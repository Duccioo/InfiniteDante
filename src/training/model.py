"""
NanoGPT Model Architecture with RoPE & Weight Tying
===================================================
Modular model definition for the InfiniteDante project.
Supports:
- Classic Learned Positional Embeddings (backward compatible with existing weights)
- Rotary Position Embeddings (RoPE) for improved relative-distance third-rhyme sensitivity
- Weight Tying between token embeddings and LM head for parameter reduction
- Configurable context window (block_size: 256, 512, 1024)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single attention head with optional Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, n_embd, head_size, block_size, dropout, use_rope=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.head_size = head_size

        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size))
            self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        if self.use_rope:
            t = torch.arange(T, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)
            half = self.head_size // 2
            q = (q * cos) + (torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin)
            k = (k * cos) + (torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head attention with projection."""
    
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout, use_rope=False):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout, use_rope=use_rope) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, n_embd, n_head, block_size, dropout, use_rope=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout, use_rope=use_rope)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    """
    NanoGPT Language Model for InfiniteDante.
    
    Supports:
    - Standard absolute learned position embeddings (use_rope=False)
    - Relative Rotary Position Embeddings (use_rope=True)
    - Weight tying (tie_weights=True)
    """
    
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=15, 
                 block_size=256, dropout=0.2, use_rope=False, tie_weights=False):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.use_rope = use_rope
        self.tie_weights = tie_weights
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not use_rope:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        else:
            self.position_embedding_table = None

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, use_rope=use_rope) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=not tie_weights)

        if tie_weights:
            self.lm_head.weight = self.token_embedding_table.weight
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        if self.position_embedding_table is not None:
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = x + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def generate_with_logprobs(self, idx, max_new_tokens, temperature=1.0):
        """Generate tokens while tracking log probabilities."""
        log_probs = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            log_prob_dist = F.log_softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            log_prob = log_prob_dist.gather(1, idx_next)
            
            log_probs.append(log_prob)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx, torch.cat(log_probs, dim=1)


class ONNXWrapper(nn.Module):
    """Wrapper for ONNX export (logits only, no loss)."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, idx):
        return self.model(idx)[0]


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())
