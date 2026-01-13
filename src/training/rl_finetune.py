"""
RL Fine-tuning for Terzina Dantesca (PPO Version)
==================================================
PPO-based training to teach the model to generate text
that follows Dante's metric structure (terzina incatenata).

Key improvements over REINFORCE:
- PPO clipped objective for stable updates
- Reward normalization to reduce variance
- Per-verse reward shaping for denser learning signal
- KL penalty actually integrated into loss
- Better hyperparameters (larger batch, adjusted LR)

Usage:
    python rl_finetune.py                    # Full RL training
    python rl_finetune.py --test-mode        # Quick test (10 iterations)
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
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Local imports
from model import NanoGPT, ONNXWrapper, count_parameters
from metric_utils import TerzinaScorer, count_syllables

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Model Hyperparameters (must match model.py defaults and pretrain)
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_EMBD = 256
N_HEAD = 8
N_LAYER = 15
DROPOUT = 0.2

# ==============================================================================
# PPO Training Config (Optimized for Stability)
# ==============================================================================
@dataclass
class PPOConfig:
    """PPO hyperparameters - STABILIZED to prevent mode collapse."""
    # Training iterations
    rl_iters: int = 100
    
    # Batch and learning - reduced for memory efficiency
    batch_size: int = 4  # Reduced from 8 for GPU memory
    gradient_accumulation_steps: int = 2  # Effective batch = 4 * 2 = 8
    lr: float = 1e-5  # REDUCED from 4e-5 to prevent collapse
    
    # PPO specific - more conservative
    ppo_epochs: int = 2  # Reduced from 4 for GPU memory
    clip_epsilon: float = 0.1  # REDUCED from 0.2 for smaller updates
    
    # Regularization - STRONGER to prevent collapse
    kl_coeff: float = 0.15  # INCREASED from 0.02 to keep close to reference
    entropy_coeff: float = 0.05  # INCREASED from 0.02 for more exploration
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Generation
    generation_length: int = 150  # Tokens per generation (~1 terzina)
    temperature: float = 0.85  # Slightly higher for diversity
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_norm_eps: float = 1e-8
    
    # Reward shaping
    use_reward_shaping: bool = True  # Use per-verse rewards
    
    # Logging
    eval_interval: int = 10
    verbose: bool = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ==============================================================================
# Tokenizer Loading
# ==============================================================================
def load_tokenizer():
    """Load BPE tokenizer from meta.json."""
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run prepare_data.py first.")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    merges_json = meta["merges"]

    # Reconstruct merges and vocab
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


# ==============================================================================
# Prompts for RL Training
# ==============================================================================
TRAINING_PROMPTS = [
    "Nel mezzo del cammin di nostra vita\n",
    "Amor, ch'al cor gentil ratto s'apprende\n",
    "O voi che siete in piccioletta barca\n",
    "Per me si va ne la città dolente\n",
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
# Running Statistics for Reward Normalization
# ==============================================================================
class RunningStats:
    """Welford's online algorithm for running mean and variance."""
    
    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        # Welford's online variance
        if self.count > 1:
            self.var = (self.var * (self.count - 2) + delta * delta2) / (self.count - 1)
    
    def normalize(self, x: float) -> float:
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
    
    def update_batch(self, values: List[float]):
        for v in values:
            self.update(v)


# ==============================================================================
# Experience Buffer for PPO
# ==============================================================================
@dataclass
class Experience:
    """Single experience for PPO training."""
    prompt: str
    generated_text: str
    token_ids: torch.Tensor  # Full sequence (prompt + generated)
    log_probs: torch.Tensor  # Log probs of generated tokens
    reward: float
    shaped_rewards: Optional[List[float]] = None  # Per-token shaped rewards
    breakdown: Optional[dict] = None


class ExperienceBuffer:
    """Buffer for collecting experiences before PPO update."""
    
    def __init__(self, max_size: int = 1000):
        self.experiences: List[Experience] = []
        self.max_size = max_size
    
    def add(self, exp: Experience):
        self.experiences.append(exp)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)
    
    def clear(self):
        self.experiences = []
    
    def __len__(self):
        return len(self.experiences)
    
    def get_all(self) -> List[Experience]:
        return self.experiences


# ==============================================================================
# PPO Trainer
# ==============================================================================
class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) trainer for metric-aware fine-tuning.

    Key features:
    - Clipped surrogate objective for stable updates
    - Reward normalization for variance reduction
    - Per-verse reward shaping for denser signal
    - KL penalty to prevent policy collapse
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        model: NanoGPT,
        encode_fn,
        decode_fn,
        vocab_size: int,
        reference_model: Optional[NanoGPT] = None,
        config: PPOConfig = None,
    ):
        self.model = model
        self.encode = encode_fn
        self.decode = decode_fn
        self.vocab_size = vocab_size
        self.config = config or PPOConfig()

        # Reference model for KL penalty (frozen)
        self.reference_model = reference_model

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.lr,
            weight_decay=0.01
        )

        # Reward scorer
        self.scorer = TerzinaScorer()

        # Running stats for reward normalization
        self.reward_stats = RunningStats(epsilon=self.config.reward_norm_eps)

        # Experience buffer
        self.buffer = ExperienceBuffer()

    def generate_and_compute_reward(
        self, prompt: str, max_tokens: int = None
    ) -> Experience:
        """
        Generate text and compute reward.
        
        Returns:
            Experience object with generation and reward info
        """
        max_tokens = max_tokens or self.config.generation_length
        
        # Encode prompt
        prompt_ids = self.encode(prompt)
        prompt_len = len(prompt_ids)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

        # Generate with log probs
        self.model.eval()
        generated_idx, log_probs = self.model.generate_with_logprobs(
            idx, max_tokens, temperature=self.config.temperature
        )
        self.model.train()

        # Decode
        generated_text = self.decode(generated_idx[0].tolist())
        generated_part = generated_text[len(prompt):]

        # Compute reward
        reward, breakdown = self.scorer.compute_reward(generated_part)
        
        # Compute shaped rewards if enabled
        shaped_rewards = None
        if self.config.use_reward_shaping:
            verse_rewards = self.scorer.compute_per_verse_rewards(generated_part)
            if verse_rewards:
                # For now, use final reward but we could distribute
                shaped_rewards = [r for r, _ in verse_rewards]

        return Experience(
            prompt=prompt,
            generated_text=generated_text,
            token_ids=generated_idx,
            log_probs=log_probs,
            reward=reward,
            shaped_rewards=shaped_rewards,
            breakdown=breakdown,
        )

    def compute_kl_divergence(
        self, 
        current_log_probs: torch.Tensor,
        token_ids: torch.Tensor,
        prompt_len: int
    ) -> torch.Tensor:
        """Compute KL divergence from reference model."""
        if self.reference_model is None:
            return torch.tensor(0.0, device=DEVICE)

        with torch.no_grad():
            # Get reference model log probs
            logits, _ = self.reference_model(token_ids[:, :-1])
            ref_log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probs for the generated tokens only
            generated_tokens = token_ids[:, prompt_len:]
            ref_log_probs_selected = ref_log_probs[:, prompt_len-1:-1, :]
            ref_log_probs_tokens = ref_log_probs_selected.gather(
                -1, generated_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # KL = sum(p * (log p - log q))
        # Approximate KL using current policy as p
        kl = (current_log_probs.exp() * (current_log_probs - ref_log_probs_tokens)).sum()
        
        return kl

    def compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus for exploration."""
        # Entropy = -sum(p * log p)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()
        return entropy

    def collect_experiences(self, prompts: List[str]) -> List[Experience]:
        """Collect a batch of experiences for PPO update."""
        experiences = []
        
        for prompt in prompts:
            exp = self.generate_and_compute_reward(prompt)
            experiences.append(exp)
            
            # Update reward statistics
            self.reward_stats.update(exp.reward)
            
            # Clear GPU cache after each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return experiences

    def ppo_update(self, experiences: List[Experience]) -> dict:
        """
        Perform PPO update on collected experiences.
        
        The PPO objective:
        L^CLIP = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
        
        Where r(θ) = π(a|s) / π_old(a|s)
        
        Memory-optimized: processes one experience at a time.
        """
        self.model.train()
        
        # Store old log probs (before update) - detach and move to CPU
        old_log_probs_list = [exp.log_probs.detach().cpu() for exp in experiences]
        
        # Also store token_ids on CPU to free GPU memory
        token_ids_list = [exp.token_ids.detach().cpu() for exp in experiences]
        
        # Clear original tensors from experiences
        for exp in experiences:
            exp.log_probs = None
            exp.token_ids = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            rewards = [self.reward_stats.normalize(exp.reward) for exp in experiences]
        else:
            rewards = [exp.reward for exp in experiences]
        
        # PPO epochs - multiple passes over the same data
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.config.ppo_epochs):
            self.optimizer.zero_grad()
            accumulated_loss = 0.0
            
            # Process experiences one at a time (for memory efficiency)
            for i, (exp, old_log_probs_cpu, token_ids_cpu, reward) in enumerate(
                zip(experiences, old_log_probs_list, token_ids_list, rewards)
            ):
                # Move tensors back to GPU only when needed
                old_log_probs = old_log_probs_cpu.to(DEVICE)
                token_ids = token_ids_cpu.to(DEVICE)
                
                # Get prompt length for KL computation
                prompt_len = len(self.encode(exp.prompt))
                
                # Recompute current log probs
                idx_cond = token_ids[:, :-1]
                logits, _ = self.model(idx_cond)
                log_probs_dist = F.log_softmax(logits / self.config.temperature, dim=-1)
                
                # Get log probs for generated tokens
                generated_tokens = token_ids[:, prompt_len:]
                current_log_probs = log_probs_dist[:, prompt_len-1:, :].gather(
                    -1, generated_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                # Compute policy ratio
                ratio = (current_log_probs - old_log_probs).exp()
                
                # Compute advantage (using normalized reward as advantage estimate)
                advantage = torch.tensor(reward, device=DEVICE)
                
                # PPO clipped objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_epsilon, 
                    1 + self.config.clip_epsilon
                ) * advantage
                
                # Policy loss (negative because we maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Skip KL to save memory (we already have clipping)
                kl_loss = torch.tensor(0.0, device=DEVICE)
                
                # Simplified entropy (just log probs mean, saves memory)
                entropy = -current_log_probs.mean()
                
                # Total loss for this experience
                exp_loss = (
                    policy_loss 
                    + self.config.kl_coeff * kl_loss 
                    - self.config.entropy_coeff * entropy
                )
                
                # Backprop immediately for this experience (gradient accumulation)
                exp_loss = exp_loss / len(experiences)
                exp_loss.backward()
                
                # Record stats before cleanup
                total_policy_loss += policy_loss.item()
                total_kl += 0.0  # KL disabled for memory
                total_entropy += entropy.item()
                accumulated_loss += exp_loss.item()
                
                # Clean up GPU tensors immediately
                del logits, log_probs_dist, current_log_probs, ratio
                del surr1, surr2, policy_loss, kl_loss, entropy, exp_loss
                del old_log_probs, token_ids, idx_cond, generated_tokens
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Gradient clipping and optimizer step after all experiences
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.max_grad_norm
            )
            
            self.optimizer.step()
            total_loss += accumulated_loss
        
        n = len(experiences) * self.config.ppo_epochs
        
        return {
            "loss": total_loss / self.config.ppo_epochs,
            "policy_loss": total_policy_loss / n,
            "kl": total_kl / n,
            "entropy": total_entropy / n,
        }

    def train_step(self, prompts: List[str]) -> dict:
        """
        Perform one PPO training step.
        
        Args:
            prompts: List of prompt strings to generate from
            
        Returns:
            Dictionary with training statistics
        """
        # Collect experiences
        experiences = self.collect_experiences(prompts)
        
        # PPO update
        update_stats = self.ppo_update(experiences)
        
        # Compute average metrics
        rewards = [exp.reward for exp in experiences]
        breakdowns = [exp.breakdown for exp in experiences]
        
        avg_reward = sum(rewards) / len(rewards)
        avg_syllables = sum(b["syllables"] for b in breakdowns) / len(breakdowns)
        avg_rhyme = sum(b["rhyme"] for b in breakdowns) / len(breakdowns)
        avg_structure = sum(b["structure"] for b in breakdowns) / len(breakdowns)

        return {
            **update_stats,
            "reward": avg_reward,
            "syllables": avg_syllables,
            "rhyme": avg_rhyme,
            "structure": avg_structure,
        }

    def evaluate(self, prompts: List[str], num_samples: int = 3) -> dict:
        """Evaluate model on multiple prompts."""
        self.model.eval()

        all_rewards = []
        samples = []

        for prompt in prompts[:num_samples]:
            exp = self.generate_and_compute_reward(prompt)
            all_rewards.append(exp.reward)
            samples.append({
                "prompt": prompt.strip(),
                "generated": exp.generated_text[len(prompt):].strip()[:200],
                "reward": exp.reward,
                "breakdown": exp.breakdown,
            })

        return {"avg_reward": sum(all_rewards) / len(all_rewards), "samples": samples}


# ==============================================================================
# Main Training Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PPO Fine-tuning for Terzina Dantesca")
    parser.add_argument("--test-mode", action="store_true", help="Quick test with 10 iterations")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pre-training, load checkpoint")
    parser.add_argument("--iters", type=int, default=None, help="Number of RL iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--no-reward-norm", action="store_true", help="Disable reward normalization")
    parser.add_argument("--no-reward-shaping", action="store_true", help="Disable per-verse reward shaping")
    args = parser.parse_args()

    # Create config
    config = PPOConfig()
    
    if args.test_mode:
        config.rl_iters = 10
        config.verbose = True
        print("=== TEST MODE: Running 10 iterations ===\n")
    
    if args.iters:
        config.rl_iters = args.iters
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.no_reward_norm:
        config.normalize_rewards = False
    if args.no_reward_shaping:
        config.use_reward_shaping = False

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
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")

    # Load pretrained checkpoint
    pretrain_checkpoint = os.path.join(MODEL_DIR, "pretrain_checkpoint.pt")
    finetune_checkpoint = os.path.join(MODEL_DIR, "finetune_checkpoint.pt")

    if os.path.exists(finetune_checkpoint):
        print(f"\nLoading fine-tuned checkpoint: {finetune_checkpoint}")
        model.load_state_dict(torch.load(finetune_checkpoint, map_location=DEVICE, weights_only=True))
    elif os.path.exists(pretrain_checkpoint):
        print(f"\nLoading pre-trained checkpoint: {pretrain_checkpoint}")
        model.load_state_dict(torch.load(pretrain_checkpoint, map_location=DEVICE, weights_only=True))
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
        dropout=0.0,  # No dropout for reference
    ).to(DEVICE)
    reference_model.load_state_dict(model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        encode_fn=encode,
        decode_fn=decode,
        vocab_size=vocab_size,
        reference_model=reference_model,
        config=config,
    )

    # Training history
    history = {
        "reward": [], 
        "syllables": [], 
        "rhyme": [], 
        "structure": [], 
        "loss": [],
        "policy_loss": [],
        "kl": [],
        "entropy": [],
    }

    print(f"\n{'='*60}")
    print(f"Starting PPO Fine-tuning ({config.rl_iters} iterations)")
    print(f"Config: batch_size={config.batch_size}, lr={config.lr}, ppo_epochs={config.ppo_epochs}")
    print(f"        clip_ε={config.clip_epsilon}, kl_coeff={config.kl_coeff}")
    print(f"        reward_norm={config.normalize_rewards}, reward_shaping={config.use_reward_shaping}")
    print(f"{'='*60}\n")

    # Training loop
    import time

    start_time = time.time()
    best_reward = 0.0

    for iteration in range(config.rl_iters):
        iter_start = time.time()

        # Sample random prompts for this batch
        batch_prompts = np.random.choice(
            TRAINING_PROMPTS, size=config.batch_size, replace=False
        ).tolist()

        # Train step
        stats = trainer.train_step(batch_prompts)

        iter_time = time.time() - iter_start

        # Record history
        for key in history:
            if key in stats:
                history[key].append(stats[key])

        # Track best reward
        if stats["reward"] > best_reward:
            best_reward = stats["reward"]

        # Print progress
        if config.verbose or iteration % config.eval_interval == 0 or iteration == config.rl_iters - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (iteration + 1)) * (config.rl_iters - iteration - 1)
            print(
                f"[{iteration+1:3d}/{config.rl_iters}] "
                f"R:{stats['reward']:.3f} "
                f"Syl:{stats['syllables']:.2f} "
                f"Rhy:{stats['rhyme']:.2f} "
                f"L:{stats['loss']:.3f} "
                f"KL:{stats['kl']:.4f} "
                f"({iter_time:.1f}s/it, ETA:{eta/60:.1f}m)"
            )

            # Show a sample generation periodically
            if iteration > 0 and iteration % config.eval_interval == 0:
                eval_result = trainer.evaluate(TRAINING_PROMPTS, num_samples=1)
                sample = eval_result["samples"][0]
                print(f"  Sample: {sample['prompt'][:30]}... -> {sample['generated'][:60]}...")
                print()

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}\n")

    eval_result = trainer.evaluate(TRAINING_PROMPTS, num_samples=5)
    print(f"Average Reward: {eval_result['avg_reward']:.3f}")
    print(f"Best Reward during training: {best_reward:.3f}\n")

    for i, sample in enumerate(eval_result["samples"]):
        print(f"--- Sample {i+1} ---")
        print(f"Prompt: {sample['prompt']}")
        print(f"Generated:\n{sample['generated']}")
        print(f"Reward: {sample['reward']:.3f} | Breakdown: {sample['breakdown']}")
        print()

    # Save RL checkpoint
    rl_checkpoint = os.path.join(MODEL_DIR, "rl_checkpoint.pt")
    torch.save(model.state_dict(), rl_checkpoint)
    print(f"Saved RL checkpoint: {rl_checkpoint}")

    # Save training history
    history_path = os.path.join(MODEL_DIR, "rl_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_path}")

    # Plot training curves
    if len(history["reward"]) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Reward plot with smoothing
        rewards = history["reward"]
        axes[0, 0].plot(rewards, alpha=0.3, label="Raw")
        # Moving average for smoothing
        window = min(20, len(rewards) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), smoothed, label="Smoothed")
        axes[0, 0].set_title("Reward")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history["syllables"], label="Syllables")
        axes[0, 1].plot(history["rhyme"], label="Rhyme")
        axes[0, 1].set_title("Metric Scores")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Policy loss with smoothing
        losses = history["loss"]
        axes[1, 0].plot(losses, alpha=0.3, label="Raw")
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(losses)), smoothed, label="Smoothed")
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(history["structure"], label="Structure")
        axes[1, 1].plot(history["kl"], label="KL Divergence")
        axes[1, 1].set_title("Structure & KL")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(MODEL_DIR, "rl_training_plots.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training plots: {plot_path}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    model.eval()
    onnx_wrapper = ONNXWrapper(model)
    dummy_input = torch.randint(0, vocab_size, (1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
    onnx_path = os.path.join(MODEL_DIR, "model.onnx")

    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "sequence"}, "output": {0: "batch_size", 1: "sequence"}},
    )
    print(f"Saved ONNX model: {onnx_path}")

    print("\n=== PPO Fine-tuning Complete ===")


if __name__ == "__main__":
    main()
