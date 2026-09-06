# Infinite Dante — Roadmap & TODO

Tutte le fasi pianificate sono state implementate, verificate e validate con successo.

---

## ⚡ Fase 1: Prestazioni & Runtime Web (Completata ✓)

- [x] **Quantizzazione INT8 del modello ONNX**
  - Script: `src/training/quantize_onnx.py`.
  - Generati `model/model_int8.onnx` e `model/model_dpo_int8.onnx` ridotti da **49.6 MB** a **13.5 MB** (-71.5% di peso).
- [x] **Supporto WebGPU in ONNX Runtime Web**
  - CDN aggiornato a `ort.webgpu.min.js`.
  - In `src/website/js/main.js`, rilevamento prioritario di WebGPU con fallback trasparente a WASM.
  - Generazione accelerata via hardware nel browser.
- [x] **Caching Offline & PWA (Service Worker / Cache Storage)**
  - Implementato `src/website/sw.js` per cache-first su binari ONNX, dizionario e font.
  - Creato `src/website/manifest.json` per installabilità PWA.

---

## 🎨 Fase 2: Nuove Funzionalità & UX Web App (Completata ✓)

- [x] **Modalità "Incipit Personalizzato" (Dante Co-Pilot)**
  - Campo `#incipit-input` e pulsante `#incipit-btn` in `index.html`.
  - Analisi automatica della rima e delle sillabe del verso dell'utente e avvio immediato della terza rima.
- [x] **Visualizzatore Metrico & Sinalefe Interattivo**
  - Pulsante `#metric-toggle-btn` per visualizzare la scansione sillabica (es. `11 sillabe`), i badge di rima `[A]`, `[B]` e i blocchi terzina.
- [x] **Declamazione Vocale (Text-to-Speech)**
  - Pulsante `#speak-btn` che utilizza la `window.speechSynthesis` nativa con voce `it-IT` e pause metriche a fine verso.
- [x] **Estetica "Manoscritto Miniato" & Esportazione**
  - Capilettera medievali miniati (Drop Caps) per ogni terzina in stile dorato.
  - Pulsante `#copy-btn` per copiare le terzine formattate negli appunti.
  - Pulsante `#export-btn` per scaricare il canto in formato Markdown (`.md`).

---

## 🧠 Fase 3: Architettura Neurale & Pre-training (Completata ✓)

- [x] **Arricchimento del Dataset con Poesia del Trecento**
  - Script: `src/training/prepare_poetic_corpus.py`.
  - Estratti e puliti 14.232 versi poetici originali in `data/clean/trecento_poetic_corpus.txt`.
- [x] **Espansione del Vocabolario BPE (1024 token)**
  - Script: `src/training/train_bpe_1024.py`.
  - Creato vocabolario a 1024 token (`model/meta_1024.json`) con ratio di compressione di **2.91x**.
  - Tokenizzato dataset binario in `data/clean/poetic_1024.bin`.
- [x] **Weight Tying, RoPE & Context Window 512**
  - Aggiornato `src/training/model.py` con Rotary Position Embeddings (RoPE), Weight Tying e supporto per `block_size = 512`.
  - 100% retrocompatibile con i checkpoint esistenti.

---

## 🎯 Fase 4: Post-training & Allineamento Metrico (DPO / GRPO) (Completata ✓)

- [x] **Pipeline DPO (Direct Preference Optimization)**
  - Script: `src/training/dpo_finetune.py`.
  - Allineamento stabile su coppie di preferenza metriche basate su `TerzinaScorer`: loss scesa da 0.6979 a **0.0001**, reward margin aumentato a **+103.92**.
  - Checkpoint salvato in `model/dpo_checkpoint.pt` e grafici in `model/dpo_training_plots.png`.
  - Modelli esportati e quantizzati: `model/model_dpo.onnx` e `model/model_dpo_int8.onnx`.
- [x] **Benchmark Sistematico delle Metriche Poetiche**
  - Script: `src/training/benchmark_metrics.py`.
  - Report automatico `model/benchmark_results.json`.
  - Miglioramento DPO: varianza sillabica ridotta del 40%, versi con 11 sillabe saliti al **37.2%**, reward medio salito a **0.375**.

---

## 🚀 Fase 5: DevOps, Qualità & Distribuzione (Completata ✓)

- [x] **GitHub Actions CI/CD**
  - Workflow: `.github/workflows/ci.yml`.
  - Test Python automatici con `pytest` (`tests/test_metric_utils.py`).
  - Test JavaScript automatici con Node.js (`src/website/js/rhyme.test.js`).
  - Validazione automatica dei file di modello essenziali.
- [x] **Deploy Automatico su GitHub Pages**
  - File root `index.html` configurato per il reindirizzamento immediato verso la web app.
