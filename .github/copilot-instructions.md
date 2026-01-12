# Infinite Dante - Copilot Instructions

## 🔭 Project Overview
Infinite Dante is a micro-LLM (~8.2M params) trained to generate Dante Alighieri-style poetry. The unique architecture involves training in **PyTorch** and running inference entirely client-side in the browser using **ONNX Runtime**.

## 🏗️ Architecture & Data Flow

### 1. Training Pipeline (Python)
- **Model Definition**: `src/training/model.py` implements a `NanoGPT` (decoder-only Transformer).
- **Tokenizer**: Custom Byte-Pair Encoding (BPE) with a small vocabulary (512 tokens) to minimize model size.
- **Workflow**:
  1. **Data Prep**: `src/training/prepare_data.py` downloads/cleans text and generates `model/meta.json`.
  2. **Training**: `src/training/train_and_export.py` runs 2-stage training (Pre-train → Fine-tune).
  3. **RL Fine-tuning**: `src/training/rl_finetune.py` uses REINFORCE to optimize for "Terzina Encatenata" metric structure.
  4. **Export**: Model is exported to ONNX format (`model/model.onnx`).

### 2. Inference Pipeline (JavaScript)
- **Engine**: ONNX Runtime Web (`onnxruntime-web`) runs `model/model.onnx`.
- **Logic**: `src/website/script.js` handles token generation loop.
- **Tokenizer**: Re-implemented in JS to match Python's BPE exactly. Uses `model/meta.json` for vocab/merges.

## 🛠️ Critical Developer Workflows

### Data Preparation
- Text cleaning in `process_directory` (`prepare_data.py`) specifically targets **Liber Liber/Project Gutenberg** headers/footers.
- **Crucial**: If `model/meta.json` changes, the JS tokenizer must utilize the new file immediately.

### Training & Export
Run from `src/training/`:
```bash
python prepare_data.py      # Generates data/clean/*.bin and model/meta.json
python train_and_export.py  # Trains and saves model/model.onnx
```

### Web Development
- No build step for frontend. Open `src/website/index.html` directly (or via local server).
- `script.js` handles **UTF-8 streaming**: `decode()` manages incomplete byte sequences across tokens.

## 🧩 Project Patterns & Conventions

- **Tokenizer Sync**: The BPE implementation in `src/training/prepare_data.py` (Python) and `src/website/script.js` (JS) must remain algorithmically identical.
- **Shared Config**: Use `model/meta.json` as the source of truth for `vocab_size`, `block_size`, and BPE merges.
- **Model Params**: Hyperparameters (n_layer=10, n_head=8, n_embd=256) are optimized for web performance (~30MB ONNX file). 
- **Path Handling**: Python scripts use `BASE_DIR` resolution to be runnable from anywhere.

## ⚠️ Gotchas
- **Vocab Size**: Strictly kept at **512** to keep embedding tables small for the browser.
- **Text Cleaning**: Aggressive regex cleaning is required for the raw text inputs to avoid learning metadata.
- **ONNX Compatibility**: Ensure custom torch modules (`NanoGPT`) remain exportable to ONNX (avoid complex dynamic control flow).
