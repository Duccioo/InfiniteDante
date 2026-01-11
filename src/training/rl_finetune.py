"""
RL Fine-tuning for Terzina Dantesca
===================================
REINFORCE-based training to teach the model to generate text
that follows Dante's metric structure (terzina incatenata).

Usage:
    python rl_finetune.py                    # Full RL training
    python rl_finetune.py --test-mode        # Quick test (5 iterations)
    python rl_finetune.py --skip-pretrain    # Skip pre-training, use checkpoint
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Local imports
from model import NanoGPT, ONNXWrapper, count_parameters
from metric_utils import TerzinaScorer, count_syllables

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'clean')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Model Hyperparameters (must match model.py defaults and pretrain)
BATCH_SIZE = 32
BLOCK_SIZE = 256
N_EMBD = 256
N_HEAD = 8
N_LAYER = 10
DROPOUT = 0.2

# RL Training Config (Optimized for CPU)
RL_ITERS = 200              # Number of RL training iterations
RL_LR = 1e-5                # Very low LR for stability
RL_BATCH_SIZE = 1           # Single sample per step (faster)
GENERATION_LENGTH = 150      # Tokens per generation (~1 terzina)
BASELINE_DECAY = 0.99       # Moving average decay for baseline
KL_COEFF = 0.01             # Penalty for diverging from reference model
EVAL_INTERVAL = 10          # How often to evaluate and print
TEMPERATURE = 0.8           # Sampling temperature
VERBOSE = True              # Print every iteration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==============================================================================
# Tokenizer Loading
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

# ==============================================================================
# Prompts for RL Training
# ==============================================================================
TRAINING_PROMPTS = [
    "Nel mezzo del cammin di nostra vita\n",
    "Amor, ch'al cor gentil ratto s'apprende\n",
    "O voi che siete in piccioletta barca\n",
    "Per me si va ne la città dolente\n",
    "Io son la via, la verità e la vita\n",
    "La gloria di colui che tutto move\n",
    "Tanto gentile e tanto onesta pare\n",
    "Donne ch'avete intelletto d'amore\n",
    "Guido, i' vorrei che tu e Lapo ed io\n",
    "Chi è questa che vèn, ch'ogn'om la mira\n",
    "Voi che per li occhi mi passaste 'l core\n",
    "Così nel mio parlar voglio esser aspro\n",
    "Io voglio del ver la mia donna laudare\n",
    "Era già l'ora che volge il disio\n",
    "Lasciate ogni speranza, voi ch'intrate\n",
    "Fatti non foste a viver come bruti\n",
    "E quindi uscimmo a riveder le stelle\n",
    "Poca favilla gran fiamma seconda\n",
    "L'amor che move il sole e l'altre stelle\n",
    "In quella parte del libro de la mia memoria\n",
]


# ==============================================================================
# RL Trainer
# ==============================================================================
class REINFORCETrainer:
    """
    REINFORCE trainer with baseline for metric-aware fine-tuning.
    
    The algorithm:
    1. Sample text from the model given prompts
    2. Compute reward using TerzinaScorer
    3. Update policy using: ∇J = E[(R - baseline) * Σ log π(a_t|s_t)]
    """
    
    def __init__(self, model, encode_fn, decode_fn, vocab_size,
                 reference_model=None, lr=RL_LR, kl_coeff=KL_COEFF):
        self.model = model
        self.encode = encode_fn
        self.decode = decode_fn
        self.vocab_size = vocab_size
        self.kl_coeff = kl_coeff
        
        # Reference model for KL penalty (optional)
        self.reference_model = reference_model
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Reward scorer
        self.scorer = TerzinaScorer()
        
        # Moving average baseline
        self.baseline = 0.0
        self.baseline_count = 0
    
    def update_baseline(self, reward: float):
        """Update moving average baseline."""
        self.baseline_count += 1
        alpha = 1.0 / min(self.baseline_count, 100)  # Faster initial adaptation
        self.baseline = (1 - alpha) * self.baseline + alpha * reward
    
    def generate_and_compute_reward(self, prompt: str, max_tokens: int = GENERATION_LENGTH
                                    ) -> Tuple[str, float, torch.Tensor]:
        """
        Generate text and compute reward.
        
        Returns:
            text: Generated text
            reward: Terzina score
            log_probs: Log probabilities of generated tokens
        """
        # Encode prompt
        prompt_ids = self.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        
        # Generate with log probs
        self.model.eval()
        generated_idx, log_probs = self.model.generate_with_logprobs(
            idx, max_tokens, temperature=TEMPERATURE
        )
        self.model.train()
        
        # Decode
        generated_text = self.decode(generated_idx[0].tolist())
        
        # Compute reward (only on generated part)
        generated_part = generated_text[len(prompt):]
        reward, breakdown = self.scorer.compute_reward(generated_part)
        
        return generated_text, reward, log_probs, breakdown
    
    def compute_kl_penalty(self, idx, log_probs):
        """Compute KL divergence penalty from reference model."""
        if self.reference_model is None:
            return 0.0
        
        with torch.no_grad():
            ref_logits, _ = self.reference_model(idx[:, :-1])
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            # Approximate KL
            kl = (log_probs.exp() * (log_probs - ref_log_probs.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1))).sum()
        
        return kl.item()
    
    def train_step(self, prompts: List[str]) -> dict:
        """
        Perform one REINFORCE training step.
        
        Args:
            prompts: List of prompt strings to generate from
            
        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        rewards = []
        breakdowns = []
        
        for prompt in prompts:
            # Generate and get reward
            text, reward, log_probs, breakdown = self.generate_and_compute_reward(prompt)
            rewards.append(reward)
            breakdowns.append(breakdown)
            
            # Compute advantage (reward - baseline)
            advantage = reward - self.baseline
            
            # REINFORCE loss: -advantage * sum of log probs
            # Negative because we want to maximize reward
            policy_loss = -advantage * log_probs.sum()
            
            total_loss += policy_loss
            
            # Update baseline
            self.update_baseline(reward)
        
        # Average loss over batch
        avg_loss = total_loss / len(prompts)
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        
        # Compute average metrics
        avg_reward = sum(rewards) / len(rewards)
        avg_syllables = sum(b['syllables'] for b in breakdowns) / len(breakdowns)
        avg_rhyme = sum(b['rhyme'] for b in breakdowns) / len(breakdowns)
        avg_structure = sum(b['structure'] for b in breakdowns) / len(breakdowns)
        
        return {
            'loss': avg_loss.item(),
            'reward': avg_reward,
            'baseline': self.baseline,
            'syllables': avg_syllables,
            'rhyme': avg_rhyme,
            'structure': avg_structure,
        }
    
    def evaluate(self, prompts: List[str], num_samples: int = 3) -> dict:
        """Evaluate model on multiple prompts."""
        self.model.eval()
        
        all_rewards = []
        samples = []
        
        for prompt in prompts[:num_samples]:
            text, reward, _, breakdown = self.generate_and_compute_reward(prompt)
            all_rewards.append(reward)
            samples.append({
                'prompt': prompt.strip(),
                'generated': text[len(prompt):].strip()[:200],
                'reward': reward,
                'breakdown': breakdown
            })
        
        return {
            'avg_reward': sum(all_rewards) / len(all_rewards),
            'samples': samples
        }


# ==============================================================================
# Main Training Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='RL Fine-tuning for Terzina Dantesca')
    parser.add_argument('--test-mode', action='store_true', help='Quick test with 5 iterations')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip pre-training, load checkpoint')
    parser.add_argument('--iters', type=int, default=RL_ITERS, help='Number of RL iterations')
    args = parser.parse_args()
    
    if args.test_mode:
        args.iters = 5
        print("=== TEST MODE: Running 5 iterations ===\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    vocab_size, encode, decode = load_tokenizer()
    print(f"Vocab size: {vocab_size}")
    
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
    
    # Load pretrained checkpoint
    pretrain_checkpoint = os.path.join(MODEL_DIR, 'pretrain_checkpoint.pt')
    finetune_checkpoint = os.path.join(MODEL_DIR, 'finetune_checkpoint.pt')
    
    if os.path.exists(finetune_checkpoint):
        print(f"\nLoading fine-tuned checkpoint: {finetune_checkpoint}")
        model.load_state_dict(torch.load(finetune_checkpoint, map_location=DEVICE))
    elif os.path.exists(pretrain_checkpoint):
        print(f"\nLoading pre-trained checkpoint: {pretrain_checkpoint}")
        model.load_state_dict(torch.load(pretrain_checkpoint, map_location=DEVICE))
    else:
        print(f"\nWARNING: No checkpoint found! Run train_and_export.py first.")
        if not args.test_mode:
            sys.exit(1)
    
    # Create reference model for KL penalty (frozen copy)
    print("Creating reference model for KL penalty...")
    reference_model = NanoGPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=0.0  # No dropout for reference
    ).to(DEVICE)
    reference_model.load_state_dict(model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Create trainer
    trainer = REINFORCETrainer(
        model=model,
        encode_fn=encode,
        decode_fn=decode,
        vocab_size=vocab_size,
        reference_model=reference_model,
        lr=RL_LR,
        kl_coeff=KL_COEFF
    )
    
    # Training history
    history = {
        'reward': [],
        'syllables': [],
        'rhyme': [],
        'structure': [],
        'loss': []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting RL Fine-tuning ({args.iters} iterations)")
    print(f"{'='*60}\n")
    
    # Training loop
    import time
    start_time = time.time()
    
    for iteration in range(args.iters):
        iter_start = time.time()
        
        # Sample random prompts for this batch
        batch_prompts = np.random.choice(TRAINING_PROMPTS, size=RL_BATCH_SIZE, replace=False).tolist()
        
        # Train step
        stats = trainer.train_step(batch_prompts)
        
        iter_time = time.time() - iter_start
        
        # Record history
        for key in history:
            if key in stats:
                history[key].append(stats[key])
        
        # Print progress (every iteration in verbose mode)
        if VERBOSE or iteration % EVAL_INTERVAL == 0 or iteration == args.iters - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (iteration + 1)) * (args.iters - iteration - 1)
            print(f"[{iteration+1:3d}/{args.iters}] "
                  f"R:{stats['reward']:.2f} "
                  f"Syl:{stats['syllables']:.2f} "
                  f"Rhy:{stats['rhyme']:.2f} "
                  f"({iter_time:.1f}s/it, ETA:{eta/60:.1f}m)")
            
            # Show a sample generation
            if iteration > 0:
                eval_result = trainer.evaluate(TRAINING_PROMPTS, num_samples=1)
                sample = eval_result['samples'][0]
                print(f"  Sample: {sample['prompt'][:30]}... -> {sample['generated'][:60]}...")
                print()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}\n")
    
    eval_result = trainer.evaluate(TRAINING_PROMPTS, num_samples=5)
    print(f"Average Reward: {eval_result['avg_reward']:.3f}\n")
    
    for i, sample in enumerate(eval_result['samples']):
        print(f"--- Sample {i+1} ---")
        print(f"Prompt: {sample['prompt']}")
        print(f"Generated:\n{sample['generated']}")
        print(f"Reward: {sample['reward']:.3f} | Breakdown: {sample['breakdown']}")
        print()
    
    # Save RL checkpoint
    rl_checkpoint = os.path.join(MODEL_DIR, 'rl_checkpoint.pt')
    torch.save(model.state_dict(), rl_checkpoint)
    print(f"Saved RL checkpoint: {rl_checkpoint}")
    
    # Save training history
    history_path = os.path.join(MODEL_DIR, 'rl_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_path}")
    
    # Plot training curves
    if len(history['reward']) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(history['reward'])
        axes[0, 0].set_title('Reward')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history['syllables'], label='Syllables')
        axes[0, 1].plot(history['rhyme'], label='Rhyme')
        axes[0, 1].set_title('Metric Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history['loss'])
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['structure'])
        axes[1, 1].set_title('Structure Score')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(MODEL_DIR, 'rl_training_plots.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training plots: {plot_path}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    model.eval()
    onnx_wrapper = ONNXWrapper(model)
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
    print(f"Saved ONNX model: {onnx_path}")
    
    print("\n=== RL Fine-tuning Complete ===")


if __name__ == '__main__':
    main()
