# 🔥 Infinite Dante

Un micro-LLM (~8.2M parametri) addestrato su testi di Dante Alighieri e letteratura italiana medievale che gira **interamente nel browser** per generare testo infinito nello stile dantesco.

![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20Web-blue)
![PyTorch](https://img.shields.io/badge/Training-PyTorch-orange)
![Parameters](https://img.shields.io/badge/Params-8.2M-green)

---

## 📖 Indice

- [Demo](#-demo)
- [Quick Start](#-quick-start)
- [Architettura](#️-architettura)
- [Training](#-training)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Dati di Training](#-dati-di-training)
- [Troubleshooting](#-troubleshooting)

---

## 🌐 Demo

Apri `src/website/index.html` con un server locale per vedere il generatore in azione con:
- **Design minimalista medievale** con font serif Cormorant Garamond
- **Controlli sempre visibili** in basso
- **Parametri LLM regolabili**: Temperature, Top-K, Top-P, Repetition Penalty
- **Modifica interattiva**: clicca sul testo quando in pausa per editarlo

---

## 🚀 Quick Start

### Prerequisiti

```bash
pip install torch numpy matplotlib
```

### 1. Prepara i Dati

```bash
# I testi vanno in due cartelle:
# - data/raw/pretraining/  → testi italiani generici
# - data/raw/finetuning/   → testi con "spirito dantesco"

cd src/training
python prepare_data.py
```

Questo crea:
- `data/clean/pretrain.bin` e `finetune.bin` (dati tokenizzati)
- `model/meta.json` (vocabolario BPE per il browser)

### 2. Addestra il Modello

```bash
cd src/training
python train_and_export.py
```

Il training include:
- **Pre-training** su letteratura italiana generale (10k iterazioni)
- **Fine-tuning** su testi danteschi (5k iterazioni)
- **Early stopping** basato sulla validation loss
- **Salvataggio automatico** di grafici e log nella cartella `model/`

Output generati in `model/`:
- `model.onnx` - Modello per inferenza browser
- `meta.json` - Vocabolario BPE
- `loss_plots.png` - Grafici loss
- `loss_history.json` - Dati loss

### 3. Avvia la Demo

```bash
# Serve un server locale per CORS (dalla root del progetto)
python -m http.server 8000

# Oppure con Node.js
npx serve .
```

Apri **http://localhost:8000/src/website/** nel browser.

---

## 🏗️ Architettura

### Modello: NanoGPT (~8.2M parametri)

| Parametro | Valore |
|-----------|--------|
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Transformer Layers | 10 |
| Context Length | 256 |
| Tokenizzazione | BPE (512 vocab) |

### Tokenizzazione BPE

- Vocabolario di 512 token (256 byte + 256 merge)
- Addestrato sul corpus combinato pretraining + finetuning
- Encoding/decoding nel browser via `meta.json`

---

## 🎓 Training

### Due Fasi

1. **Pre-training** (General Italian)
   - Learning rate: 3e-4
   - 10,000 iterazioni
   - Dati: letteratura italiana medievale e classica

2. **Fine-tuning** (Dante)
   - Learning rate: 1e-4
   - 5,000 iterazioni
   - Dati: opere di Dante e testi con "spirito dantesco"

### Early Stopping

- **Patience**: 5 valutazioni consecutive senza miglioramento
- Salva automaticamente il miglior modello
- Ripristina i pesi migliori a fine training

### Visualizzazione Loss

Dopo il training vengono generati in `model/`:
- `loss_plots.png` - Grafici di train/val loss per entrambe le fasi
- `loss_history.json` - Dati raw per analisi successive

---

## 🎛️ Controlli UI

| Controllo | Descrizione |
|-----------|-------------|
| **Temperature** | Creatività (0.3 = conservativo, 1.5 = creativo) |
| **Top-K** | Considera solo i K token più probabili |
| **Top-P** | Nucleus sampling (probabilità cumulativa) |
| **Rep Penalty** | Penalizza ripetizioni (>1.0 = meno ripetizioni) |
| **Speed** | Ritardo tra caratteri (ms) |
| **Spazio** | Play/Pausa generazione |
| **Escape** | Stop generazione |

### Modifica Interattiva

Quando in pausa, clicca sul testo per modificarlo. Il modello continuerà dalle tue modifiche.

---

## 📁 Struttura del Progetto

```
InfiniteDante/
├── data/
│   ├── raw/
│   │   ├── pretraining/         # Testi italiani generici (.txt)
│   │   └── finetuning/          # Testi danteschi (.txt)
│   └── clean/
│       ├── pretrain.bin         # Token pretraining (generato)
│       └── finetune.bin         # Token finetuning (generato)
├── model/
│   ├── meta.json                # Vocabolario BPE (generato)
│   ├── model.onnx               # Modello trainato (generato)
│   ├── loss_plots.png           # Grafici loss (generato)
│   └── loss_history.json        # Storia loss (generato)
├── src/
│   ├── training/
│   │   ├── prepare_data.py      # Preprocessing e BPE tokenizer
│   │   └── train_and_export.py  # Training con early stopping + export ONNX
│   └── website/
│       ├── index.html           # UI web (design medievale minimalista)
│       └── script.js            # Inferenza browser + sampling avanzato
├── LICENSE                      # CC BY-NC 4.0
└── README.md
```

---

## 📚 Dati di Training

### Finetuning (Spirito Dantesco)

| Opera | Autore | Link |
|-------|--------|------|
| La Divina Commedia | Dante | [Gutenberg](https://www.gutenberg.org/cache/epub/8800/pg8800.txt) |
| Convivio | Dante | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001673) |
| Rime | Dante | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000691) |
| De Vulgari Eloquentia | Dante | [Archive.org](https://archive.org/stream/iltrattatodevulgarielomino/iltrattatodevulgarielomino_djvu.txt) |
| Vita Nuova | Dante | [Archive.org](https://archive.org/stream/vitanuovadantealighieri/Vita%20Nuova%20-%20Dante%20Alighieri_djvu.txt) |
| Epistole | Dante | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000312) |

### Pretraining (Italiano Medievale/Classico)

| Opera | Autore | Link |
|-------|--------|------|
| Promessi Sposi | Manzoni | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000267) |
| Il Principe | Machiavelli | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000214) |
| Rime | Cecco Angiolieri | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000728) |
| Commedia delle Ninfe | Boccaccio | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000738) |
| Chiose all'Inferno | Jacopo Alighieri | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000221) |
| Amorosa Visione | Boccaccio | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000045) |
| Teseida | Boccaccio | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000041) |
| L'Acerba | Cecco d'Ascoli | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001436) |
| Poesie | Cino da Pistoia | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001110) |
| La Spagna | Anonimo | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000653) |
| Rime | Antonio Beccari | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit001342) |
| Poesie | Antonio degli Agli | [Biblioteca Italiana](http://www.bibliotecaitaliana.it/testo/bibit000542) |

---

## 🐛 Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| Errore CORS | Usa un server locale (`python -m http.server`) |
| "Failed to load meta.json" | Esegui prima `prepare_data.py` poi `train_and_export.py` |
| Generazione lenta | Abbassa temperature, usa browser più veloce |
| Testo ripetitivo | Aumenta Rep Penalty, abbassa temperature |
| Early stopping troppo presto | Aumenta `EARLY_STOPPING_PATIENCE` in `train_and_export.py` |
| Memoria GPU insufficiente | Riduci `BATCH_SIZE` o `N_EMBD` |

---

## 📜 Licenza

Progetto educativo. La Divina Commedia e gli altri testi medievali sono di pubblico dominio.

## 🙏 Riconoscimenti

- [nanoGPT di Andrej Karpathy](https://github.com/karpathy/nanoGPT) - Ispirazione architetturale
- [ONNX Runtime Web](https://onnxruntime.ai/) - Inferenza nel browser
- [Biblioteca Italiana](http://www.bibliotecaitaliana.it/) - Testi digitalizzati