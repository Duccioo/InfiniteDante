# 🔥 Infinite Dante

A micro-LLM (~8.2M parameters) trained on Dante Alighieri's works and medieval Italian literature that runs **entirely in the browser** to generate infinite Dante-style text.

![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20Web-blue)
![PyTorch](https://img.shields.io/badge/Training-PyTorch-orange)
![Parameters](https://img.shields.io/badge/Params-8.2M-green)

---

## 📖 Table of Contents

- [Demo](#-demo)
- [Quick Start](#-quick-start)
- [Architecture](#️-architecture)
- [Training](#-training)
- [Project Structure](#-project-structure)
- [Training Data](#-training-data)
- [Troubleshooting](#-troubleshooting)

---

## 🌐 Demo

Open `src/website/index.html` with a local server to see the generator in action:
- **Minimalist medieval design** with Cormorant Garamond serif font
- **Always-visible controls** at the bottom
- **Adjustable LLM parameters**: Temperature, Top-K, Top-P, Repetition Penalty
- **Interactive editing**: click on text when paused to edit it

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib
```

### 1. Prepare the Data

```bash
# Text files go in two folders:
# - data/raw/pretraining/  → general Italian texts
# - data/raw/finetuning/   → texts with "Dantean spirit"

cd src/training
python prepare_data.py
```

This creates:
- `data/clean/pretrain.bin` and `finetune.bin` (tokenized data)
- `model/meta.json` (BPE vocabulary for browser)

### 2. Train the Model

```bash
cd src/training
python train_and_export.py
```

Training includes:
- **Pre-training** on general Italian literature (10k iterations)
- **Fine-tuning** on Dante's works (2k iterations)
- **Early stopping** based on validation loss
- **Automatic saving** of plots and logs to `model/` folder

Generated outputs in `model/`:
- `model.onnx` - Model for browser inference
- `meta.json` - BPE vocabulary
- `loss_plots.png` - Loss charts
- `loss_history.json` - Loss data

### 3. Run the Demo

```bash
# Start a local server for CORS (from project root)
python -m http.server 8000

# Or with Node.js
npx serve .
```

Open **http://localhost:8000** in your browser.

---

## 🏗️ Architecture

### Model: NanoGPT (~8.2M parameters)

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Transformer Layers | 10 |
| Context Length | 256 |
| Tokenization | BPE (512 vocab) |

### BPE Tokenization

- 512-token vocabulary (256 bytes + 256 merges)
- Trained on combined pretraining + finetuning corpus
- Browser encoding/decoding via `meta.json`

---

## 🎓 Training

### Two Phases

1. **Pre-training** (General Italian)
   - Learning rate: 3e-4
   - 10,000 iterations
   - Data: medieval and classical Italian literature

2. **Fine-tuning** (Dante)
   - Learning rate: 5e-5
   - 2,000 iterations
   - Data: Dante's works and texts with "Dantean spirit"

### Early Stopping

- **Patience**: 5 evals for pretraining, 10 for finetuning
- Automatically saves the best model
- Restores best weights at training end

### Loss Visualization

After training, generated in `model/`:
- `loss_plots.png` - Train/val loss charts for both phases
- `loss_history.json` - Raw data for further analysis

---

## 🎛️ UI Controls

| Control | Description |
|---------|-------------|
| **Temperature** | Creativity (0.3 = conservative, 1.5 = creative) |
| **Top-K** | Consider only the K most likely tokens |
| **Top-P** | Nucleus sampling (cumulative probability) |
| **Rep Penalty** | Penalize repetitions (>1.0 = fewer repetitions) |
| **Speed** | Delay between characters (ms) |
| **Space** | Play/Pause generation |
| **Escape** | Stop generation |

### Interactive Editing

When paused, click on the text to edit it. The model will continue from your modifications.

---

## 📁 Project Structure

```
InfiniteDante/
├── data/
│   ├── raw/
│   │   ├── pretraining/         # General Italian texts (.txt)
│   │   └── finetuning/          # Dante-style texts (.txt)
│   └── clean/
│       ├── pretrain.bin         # Pretraining tokens (generated)
│       └── finetune.bin         # Finetuning tokens (generated)
├── model/
│   ├── meta.json                # BPE vocabulary (generated)
│   ├── model.onnx               # Trained model (generated)
│   ├── loss_plots.png           # Loss charts (generated)
│   └── loss_history.json        # Loss history (generated)
├── src/
│   ├── training/
│   │   ├── prepare_data.py      # Preprocessing & BPE tokenizer
│   │   └── train_and_export.py  # Training with early stopping + ONNX export
│   └── website/
│       ├── index.html           # Web UI (minimalist medieval design)
│       └── script.js            # Browser inference + advanced sampling
├── LICENSE                      # CC BY-NC 4.0
└── README.md
```

---

## 📚 Training Data

### Finetuning (Dantean Spirit)

| Work | Author | Link |
|------|--------|------|
| The Divine Comedy | Dante | [Gutenberg](https://www.gutenberg.org/cache/epub/8800/pg8800.txt) |
| Convivio | Dante | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001673) |
| Rime | Dante | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000691) |
| De Vulgari Eloquentia | Dante | [Archive.org](https://archive.org/stream/iltrattatodevulgarielomino/iltrattatodevulgarielomino_djvu.txt) |
| Vita Nuova | Dante | [Archive.org](https://archive.org/stream/vitanuovadantealighieri/Vita%20Nuova%20-%20Dante%20Alighieri_djvu.txt) |
| Epistole | Dante | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000312) |

### Pretraining (Medieval/Classical Italian)

| Work | Author | Link |
|------|--------|------|
| The Betrothed | Manzoni | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000267) |
| The Prince | Machiavelli | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000214) |
| Rime | Cecco Angiolieri | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000728) |
| Comedy of the Nymphs | Boccaccio | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000738) |
| Glosses on Inferno | Jacopo Alighieri | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000221) |
| Amorosa Visione | Boccaccio | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000045) |
| Teseida | Boccaccio | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000041) |
| L'Acerba | Cecco d'Ascoli | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001436) |
| Poems | Cino da Pistoia | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001110) |
| La Spagna | Anonymous | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000653) |
| Rime | Antonio Beccari | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001342) |
| Poems | Antonio degli Agli | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000542) |

---

## 🌍 Deploy to GitHub Pages

### 1. Push the Repository

```bash
git add .
git commit -m "Deploy Infinite Dante"
git push origin main
```

### 2. Enable GitHub Pages

1. Go to **GitHub → Repository → Settings**
2. Click **Pages** in the sidebar
3. Under **Source** select:
   - **Branch**: `main`
   - **Folder**: `/ (root)`
4. Click **Save**

### 3. Access the Site

After a few minutes, the site will be available at:
```
https://YOUR-USERNAME.github.io/InfiniteDante/
```

The `index.html` in root automatically redirects to `src/website/`.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| CORS error | Use a local server (`python -m http.server`) |
| "Failed to load meta.json" | Run `prepare_data.py` first, then `train_and_export.py` |
| Slow generation | Lower temperature, use a faster browser |
| Repetitive text | Increase Rep Penalty, lower temperature |
| Early stopping too soon | Increase `EARLY_STOPPING_PATIENCE` in `train_and_export.py` |
| GPU out of memory | Reduce `BATCH_SIZE` or `N_EMBD` |

---

## 📜 License

Educational project. The Divine Comedy and other medieval texts are in the public domain.

## 🙏 Acknowledgments

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Architectural inspiration
- [ONNX Runtime Web](https://onnxruntime.ai/) - Browser inference
- [Biblioteca Italiana](http://www.bibliotecaitaliana.it/) - Digitized texts