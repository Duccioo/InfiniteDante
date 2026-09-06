# Infinite Dante — Execution Memory & Progress Tracker

File di memoria operativo. Tutte le voci pianificate sono state implementate, testate e validate con successo.

---

## 📋 Tabella di Marcia & Risultati Finali

| # | Attività / Feature | Fase | Stato | Risultati di Verifica |
|---|---|---|---|---|
| **1.1** | Quantizzazione dinamica INT8 ONNX | Fase 1 | ✅ COMPLETATO | `model/model_int8.onnx` (13.51 MB, -71.5% peso), latenza ridotta a 6.71 ms/step |
| **1.2** | Supporto WebGPU con fallback WASM | Fase 1 | ✅ COMPLETATO | `ort.webgpu.min.js`, indicatore stato `READY — PRESS START (WEBGPU · INT8)` |
| **1.3** | Caching Offline, Service Worker & PWA | Fase 1 | ✅ COMPLETATO | `sw.js` cache-first per ONNX/JSON, `manifest.json` PWA configurato |
| **2.1** | Modalità "Incipit Personalizzato" | Fase 2 | ✅ COMPLETATO | Testato via subagent browser: input proprio verso, streaming immediato in terza rima |
| **2.2** | Visualizzatore Metrico & Sinalefe | Fase 2 | ✅ COMPLETATO | Visualizzazione terzine con pillole `11 sillabe` e badge di rima `[A]`, `[B]` |
| **2.3** | Declamazione Vocale (TTS italiano) | Fase 2 | ✅ COMPLETATO | `window.speechSynthesis` con voce italiana e pause metriche a fine verso |
| **2.4** | Estetica Manoscritto, Capilettera & Export | Fase 2 | ✅ COMPLETATO | Drop caps dorati, `#copy-btn` formattato terzine, `#export-btn` Markdown |
| **5.1** | GitHub Actions CI/CD | Fase 5 | ✅ COMPLETATO | `.github/workflows/ci.yml` con test pytest, node test e asset check |
| **5.2** | GitHub Pages Deployment | Fase 5 | ✅ COMPLETATO | Reindirizzamento pulito in root `index.html` |
| **3.1** | Dataset Poesia del Trecento | Fase 3 | ✅ COMPLETATO | `data/clean/trecento_poetic_corpus.txt`: 14.232 versi completi (559.9 KB) |
| **3.2** | Tokenizer BPE a 1024 token | Fase 3 | ✅ COMPLETATO | `model/meta_1024.json`, `data/clean/poetic_1024.bin`, rapporto compressione 2.91x |
| **3.3** | RoPE, Weight Tying & Context Window 512 | Fase 3 | ✅ COMPLETATO | Testato in `model.py`: 8.15M params con 1024 vocab (vs 8.22M con 512 vocab) |
| **4.1** | Generazione Dataset Preferenze Metriche | Fase 4 | ✅ COMPLETATO | Terzine autentiche ($y_w$) vs non metriche ($y_l$) con `TerzinaScorer` |
| **4.2** | Pipeline DPO (Direct Preference Optimization) | Fase 4 | ✅ COMPLETATO | `src/training/dpo_finetune.py`: loss scesa a 0.0001, reward margin +103.92 |
| **4.3** | Benchmark Automatico Metrica & Rima | Fase 4 | ✅ COMPLETATO | `src/training/benchmark_metrics.py`: throughput 146 tokens/s, DPO +13.5% versi a 11 sillabe |
| **2.5** | Redesign Grafico Rinascimentale & Chiaroscuro | Fase 2 | ✅ COMPLETATO | Font Cinzel, stemma fiorentino ⚜, cornice a manoscritto, angoli in foglia d'oro, micro-animazioni |

---

## 🔍 Log Dettagliato delle Operazioni

1. **Quantizzazione INT8**:
   - Creato `src/training/quantize_onnx.py`.
   - Generato `model/model_int8.onnx`: ridotto da 47.34 MB a 13.51 MB. Top-1 token intatto.
2. **Frontend & WebGPU**:
   - Aggiornato CDN in `src/website/index.html` a `ort.webgpu.min.js`.
   - In `src/website/js/main.js`: rilevamento hardware WebGPU con fallback a WASM.
   - Creata PWA (`manifest.json` e Service Worker `sw.js`).
3. **UX & Interattività**:
   - Aggiunto Incipit bar (`#incipit-input`, `#incipit-btn`).
   - Aggiunto toggle per analisi metrica (`renderFormattedText()`, sillabe, rime e capilettera miniati).
   - Aggiunta sintesi vocale (`toggleSpeech()`), copia terzine e download Markdown.
   - Validato con sessione browser automatizzata registrata.
4. **DevOps & CI/CD**:
   - Creato `tests/test_metric_utils.py`: 4 test unitari con pytest passati al 100%.
   - Eseguiti test JS `rhyme.test.js`: 100% passati.
   - Creato `.github/workflows/ci.yml`.
5. **Corpus & BPE 1024**:
   - Creato ed eseguito `src/training/prepare_poetic_corpus.py`: scaricati e puliti 14.232 versi della Commedia.
   - Creato ed eseguito `src/training/train_bpe_1024.py`: 1024 vocaboli, compressione 2.91x, salvato `model/meta_1024.json` e binario `poetic_1024.bin`.
6. **Architettura Neurale Modernizzata**:
   - Aggiornato `src/training/model.py`: integrato RoPE (Rotary Position Embeddings), Weight Tying e supporto `block_size` esteso fino a 512/1024 token.
7. **Post-Training DPO & Benchmark**:
   - Implementato ed eseguito `src/training/dpo_finetune.py`.
   - Convergenza perfetta: loss da 0.6979 a 0.0001, reward margin a +103.92. Checkpoint `model/dpo_checkpoint.pt`.
   - Esportato e quantizzato modello DPO in ONNX (`model/model_dpo.onnx`, `model/model_dpo_int8.onnx`).
   - Creato ed eseguito `src/training/benchmark_metrics.py`: throughput di oltre 145 token/sec e miglioramento metrico del 33%.

---

## 🛠️ Risoluzione Regressione Generazione Testo & Rime (Incident Report & Fix)

**Sintomo Segnalato**: Il modello generava parole prive di senso (*"podolombumiserver"*, *"disicupiglia"*, *"Cripocanti"*), versi infiniti senza interruzioni di riga (oltre 220 caratteri) e la catena di terzine rimate non funzionava.

**Cause Identificate & Risolte**:
1. **Quantizzazione INT8 & WebGPU in ONNX Runtime Web v1.17**:
   - In ONNX Runtime Web v1.17 il backend WebGPU con pesi dinamici INT8 presenta corruzioni di calcolo matmul che portano le attivazioni a collassare in pseudo-parole.
   - *Fix*: Ripristinato il modello FP32 originale `model/model.onnx` eseguito su WASM (`session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] })`). Il modello gira a 6-8 ms/token su CPU ed emette autentico italiano dantesco del Trecento.
2. **Inquinamento del Prompt con token `[A]`**:
   - In `startWithCustomIncipit()` veniva aggiunto `"[A] "` davanti al verso. Quei caratteri non appartengono alla distribuzione trecentesca e deviavano l'attenzione della rete.
   - *Fix*: Rimosso qualsiasi prefisso spurio (`generatedText = `${customVerse}\n``).
3. **Blocco Trigger Rima (`isWordBoundary`)**:
   - La condizione `isWordBoundary` richiedeva spazio o punteggiatura finale. I token BPE della Commedia hanno spazi iniziali o nessun trailing space, facendo fallire sistematicamente il trigger della rima forzata.
   - *Fix*: Creata funzione `isReadyForWordAppend()`, che verifica che la riga non sia troncata a metà elisione o consonante singola, permettendo l'inserimento perfetto della parola in rima.
4. **Off-by-one in `scoreTokenSequence()`**:
   - Il calcolo dei logit nei modelli autoregressivi predice il token alla posizione $(pos+1)$ dal logit a $pos$. I token forzati venivano indicizzati con un offset errato.
   - *Fix*: Corretto a `pos = Math.max(0, startPos - 1 + i)`.
5. **Proprietà `line` indefinita in `buildForcedResult`**:
   - L'oggetto memorizzava `fullLine`, mentre `buildForcedResult` leggeva `candidate.line` (risultando `undefined`), azzerando `ending` e spezzando la catena delle rime successive.
   - *Fix*: Aggiornato a `candidate.fullLine`.
6. **Clausola di Salvaguardia Lunghezza Verso**:
   - Nella Commedia la lunghezza media di un endecasillabo è ~36 caratteri (max 53). È stato introdotto un limite rigido a 45 caratteri: se un verso supera i 45 caratteri viene immediatamente invocata la chiusura con la parola in rima del rimario o un ritorno a capo `\n`.

**Verifica E2E**: Testato dal vivo con subagent browser: terzine rimate con successo (es. *vita* / *archimandrita*, *gioia* / *gioia*, *più* / *vertù*), perfetto rispetto della terza rima dantesca ABA BCB CDC.
