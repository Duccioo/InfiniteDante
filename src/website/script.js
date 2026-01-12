/**
 * script.js
 * =========
 * Infinite Dante - Browser-based inference for the Micro-LLM Transformer.
 * Neo-Brutalist Terminal Edition with advanced sampling.
 */

// ============================================================================
// Global State
// ============================================================================
let session = null;          // ONNX inference session
let meta = null;             // Tokenizer metadata { merges, vocab_size, block_size }
let isGenerating = false;    // Generation loop control
let generatedText = '';      // Accumulated generated text
let cantoCount = 1;          // Current canto number
let charsSinceLastCanto = 0; // Characters since last canto break
const CHARS_PER_CANTO = 800; // Characters before inserting a canto break

// Sampling Parameters (with defaults)
let temperature = 0.85;
let topK = 40;
let topP = 0.92;
let repetitionPenalty = 1.15;
let speed = 50;
let showContextWindow = false;
let currentTokens = [];

// DOM Elements (initialized after DOMContentLoaded)
let textContainer, textOutput, cursorEl;
let startBtn, stopBtn, clearBtn;
let temperatureSlider, topKSlider, topPSlider, repPenaltySlider, speedSlider;
let tempValue, topKValue, topPValue, repPenaltyValue, speedValue;
let statusEl, editIndicator;
let contextToggleBtn, contextWindowDisplay, ctxContent, ctxTokenCount;

// ============================================================================
// Tokenizer Functions (BPE)
// ============================================================================

let bpe_vocab = []; // Array of Uint8Arrays where index is token ID

/**
 * Initialize the BPE vocabulary from merges.
 */
function initBPE(merges_json) {
    bpe_vocab = Array.from({ length: 256 }, (_, i) => new Uint8Array([i]));
    const merges = Object.entries(merges_json).sort((a, b) => a[1] - b[1]);
    for (const [pair_str, idx] of merges) {
        const [p0, p1] = pair_str.split(',').map(Number);
        const combined = new Uint8Array(bpe_vocab[p0].length + bpe_vocab[p1].length);
        combined.set(bpe_vocab[p0]);
        combined.set(bpe_vocab[p1], bpe_vocab[p0].length);
        bpe_vocab[idx] = combined;
    }
}

/**
 * Encode a string to BPE token IDs.
 */
function encode(str) {
    let tokens = Array.from(new TextEncoder().encode(str));
    const merges = Object.entries(meta.merges).sort((a, b) => a[1] - b[1]);
    for (const [pair_str, idx] of merges) {
        const [p0, p1] = pair_str.split(',').map(Number);
        let i = 0;
        const next_tokens = [];
        while (i < tokens.length) {
            if (i < tokens.length - 1 && tokens[i] === p0 && tokens[i + 1] === p1) {
                next_tokens.push(Number(idx));
                i += 2;
            } else {
                next_tokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = next_tokens;
    }
    return tokens;
}

/**
 * Decode BPE token IDs back to a string.
 * Handles incomplete UTF-8 sequences by buffering trailing bytes.
 */
let pendingBytes = new Uint8Array(0); // Buffer for incomplete UTF-8 sequences

function decode(tokens) {
    // Calculate total length including pending bytes
    const tokenLength = tokens.reduce((acc, t) => acc + (bpe_vocab[t] ? bpe_vocab[t].length : 0), 0);
    const result = new Uint8Array(pendingBytes.length + tokenLength);

    // Copy pending bytes first
    result.set(pendingBytes, 0);
    let offset = pendingBytes.length;

    // Add new token bytes
    for (const t of tokens) {
        if (bpe_vocab[t]) {
            result.set(bpe_vocab[t], offset);
            offset += bpe_vocab[t].length;
        }
    }

    // Find the last complete UTF-8 character boundary
    let validEnd = result.length;
    for (let i = Math.max(0, result.length - 4); i < result.length; i++) {
        const byte = result[i];
        // Check if this is a UTF-8 start byte
        if ((byte & 0x80) === 0) {
            // ASCII - always complete
            validEnd = i + 1;
        } else if ((byte & 0xE0) === 0xC0) {
            // 2-byte sequence start
            if (i + 2 <= result.length) validEnd = i + 2;
            else { validEnd = i; break; }
        } else if ((byte & 0xF0) === 0xE0) {
            // 3-byte sequence start
            if (i + 3 <= result.length) validEnd = i + 3;
            else { validEnd = i; break; }
        } else if ((byte & 0xF8) === 0xF0) {
            // 4-byte sequence start
            if (i + 4 <= result.length) validEnd = i + 4;
            else { validEnd = i; break; }
        }
    }

    // Store incomplete bytes for next call
    if (validEnd < result.length) {
        pendingBytes = result.slice(validEnd);
    } else {
        pendingBytes = new Uint8Array(0);
    }

    // Decode only the complete portion
    const completeBytes = result.slice(0, validEnd);
    return new TextDecoder('utf-8', { fatal: false }).decode(completeBytes);
}

/**
 * Reset pending bytes buffer (call when clearing text)
 */
function resetDecoder() {
    pendingBytes = new Uint8Array(0);
}

// ============================================================================
// Advanced Sampling Functions
// ============================================================================

/**
 * Apply Top-K filtering to logits.
 * Only keep the top K highest logits, set others to -Infinity.
 */
function applyTopK(logits, k) {
    if (k <= 0 || k >= logits.length) return logits;

    const indexed = Array.from(logits).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);

    const topIndices = new Set(indexed.slice(0, k).map(x => x.i));
    return logits.map((v, i) => topIndices.has(i) ? v : -Infinity);
}

/**
 * Apply repetition penalty to logits.
 * Reduce probability of tokens that appeared recently.
 */
function applyRepetitionPenalty(logits, recentTokens, penalty) {
    if (penalty <= 1.0) return logits;

    const penalized = Float32Array.from(logits);
    const recentSet = new Set(recentTokens);

    for (const token of recentSet) {
        if (token >= 0 && token < penalized.length) {
            if (penalized[token] > 0) {
                penalized[token] /= penalty;
            } else {
                penalized[token] *= penalty;
            }
        }
    }
    return penalized;
}

/**
 * Compute softmax with temperature scaling.
 */
function softmax(logits, temp = 1.0) {
    const scaled = Array.from(logits).map(x => x / temp);
    const max = Math.max(...scaled.filter(x => x !== -Infinity));
    const exp = scaled.map(x => x === -Infinity ? 0 : Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
}

/**
 * Apply Top-P (nucleus) sampling.
 * Sample from the smallest set of tokens whose cumulative probability >= p.
 */
function applyTopP(probs, p) {
    if (p >= 1.0) return probs;

    const indexed = probs.map((prob, idx) => ({ prob, idx }));
    indexed.sort((a, b) => b.prob - a.prob);

    let cumulative = 0;
    const nucleus = new Set();

    for (const item of indexed) {
        cumulative += item.prob;
        nucleus.add(item.idx);
        if (cumulative >= p) break;
    }

    // Re-normalize probabilities within nucleus
    const filtered = probs.map((p, i) => nucleus.has(i) ? p : 0);
    const sum = filtered.reduce((a, b) => a + b, 0);
    return filtered.map(p => p / sum);
}

/**
 * Sample from a probability distribution.
 */
function sample(probs) {
    const rand = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (rand < cumulative) return i;
    }
    return probs.length - 1;
}

// ============================================================================
// Model Inference
// ============================================================================

/**
 * Run inference on the ONNX model.
 */
async function runInference(tokens) {
    const inputTokens = tokens.slice(-meta.block_size);
    const seqLen = inputTokens.length;
    const inputArray = new BigInt64Array(inputTokens.map(t => BigInt(t)));
    const inputTensor = new ort.Tensor('int64', inputArray, [1, seqLen]);
    const results = await session.run({ input: inputTensor });
    const output = results.output;
    const vocabSize = meta.vocab_size;
    const lastIdx = (seqLen - 1) * vocabSize;
    return output.data.slice(lastIdx, lastIdx + vocabSize);
}

/**
 * Generate the next token with advanced sampling.
 */
async function generateNext(context) {
    let logits = await runInference(context);

    // Apply repetition penalty (look at last 64 tokens)
    const recentTokens = context.slice(-64);
    logits = applyRepetitionPenalty(logits, recentTokens, repetitionPenalty);

    // Apply Top-K filtering
    logits = applyTopK(logits, topK);

    // Convert to probabilities with temperature
    let probs = softmax(logits, temperature);

    // Apply Top-P (nucleus) filtering
    probs = applyTopP(probs, topP);

    return sample(probs);
}

// ============================================================================
// Text Display & Canto Management
// ============================================================================

/**
 * Create a canto separator element.
 */
function createCantoSeparator(cantoNum) {
    const separator = document.createElement('div');
    separator.className = 'canto-separator';
    separator.textContent = `CANTO ${toRoman(cantoNum)}`;
    return separator;
}

/**
 * Convert number to Roman numerals.
 */
function toRoman(num) {
    const romanNumerals = [
        ['M', 1000], ['CM', 900], ['D', 500], ['CD', 400],
        ['C', 100], ['XC', 90], ['L', 50], ['XL', 40],
        ['X', 10], ['IX', 9], ['V', 5], ['IV', 4], ['I', 1]
    ];
    let result = '';
    for (const [letter, value] of romanNumerals) {
        while (num >= value) {
            result += letter;
            num -= value;
        }
    }
    return result;
}

/**
 * Update the display with new text.
 */
function updateDisplay(newChar) {
    generatedText += newChar;
    charsSinceLastCanto += newChar.length;

    // Check for canto break
    if (charsSinceLastCanto >= CHARS_PER_CANTO && (newChar === '.' || newChar === '\n')) {
        // Insert canto break
        cantoCount++;
        charsSinceLastCanto = 0;

        // Add separator to DOM
        const separator = createCantoSeparator(cantoCount);
        textContainer.insertBefore(separator, cursorEl);
    }

    // Update text content
    textOutput.textContent = generatedText;

    // Smart auto-scroll: only if user is near the bottom (within 100px)
    const scrollBottom = textContainer.scrollHeight - textContainer.scrollTop - textContainer.clientHeight;
    if (scrollBottom < 100) {
        textContainer.scrollTop = textContainer.scrollHeight;
    }
}

/**
 * Render the full text with canto breaks.
 */
function renderFullText() {
    // Clear existing separators
    textContainer.querySelectorAll('.canto-separator').forEach(el => el.remove());
    textOutput.textContent = generatedText;
}

/**
 * Update the context window display with current tokens.
 */
function updateContextWindowDisplay(tokens, newTokenCount = 1) {
    if (!showContextWindow || !ctxContent) return;

    // Get the context window tokens (last block_size tokens)
    const contextTokens = tokens.slice(-meta.block_size);
    
    // Decode all tokens to get the text
    const allBytes = [];
    for (const t of contextTokens) {
        if (bpe_vocab[t]) {
            allBytes.push(...bpe_vocab[t]);
        }
    }
    
    // Decode to text
    const contextText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(allBytes));
    
    // Calculate where the new tokens start in the text
    const oldTokens = contextTokens.slice(0, -newTokenCount);
    const oldBytes = [];
    for (const t of oldTokens) {
        if (bpe_vocab[t]) {
            oldBytes.push(...bpe_vocab[t]);
        }
    }
    const oldText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(oldBytes));
    
    // Split text into old and new parts
    const oldPart = contextText.slice(0, oldText.length);
    const newPart = contextText.slice(oldText.length);
    
    // Update display with highlighted new token
    ctxContent.innerHTML = escapeHtml(oldPart) + 
        (newPart ? `<span class="ctx-new">${escapeHtml(newPart)}</span>` : '');
    
    // Update token count
    ctxTokenCount.textContent = `${contextTokens.length} / ${meta.block_size} tokens`;
    
    // Auto-scroll to bottom
    contextWindowDisplay.scrollTop = contextWindowDisplay.scrollHeight;
}

/**
 * Escape HTML characters for safe display.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Interactive Editing
// ============================================================================

/**
 * Enable text editing mode.
 */
function enableEditing() {
    textOutput.contentEditable = 'true';
    textOutput.classList.add('editable');
    editIndicator.classList.add('visible');
    cursorEl.style.display = 'none';
}

/**
 * Disable text editing mode.
 */
function disableEditing() {
    textOutput.contentEditable = 'false';
    textOutput.classList.remove('editable');
    editIndicator.classList.remove('visible');
    cursorEl.style.display = 'inline-block';

    // Sync text content back
    generatedText = textOutput.textContent;
}

// ============================================================================
// Generation Loop
// ============================================================================

async function startGeneration() {
    if (isGenerating) return;

    isGenerating = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = 'GENERATING...';
    statusEl.classList.add('active');

    disableEditing();

    // Initialize with seed text if empty
    if (generatedText.length === 0) {
        generatedText = 'Nel mezzo del cammin di nostra vita ';
        textOutput.textContent = generatedText;
        charsSinceLastCanto = generatedText.length;
    }

    // Convert current text to tokens
    let tokens = encode(generatedText);
    if (tokens.length === 0) tokens = [0];

    // Generation loop
    while (isGenerating) {
        try {
            const nextToken = await generateNext(tokens);
            tokens.push(nextToken);

            if (tokens.length > meta.block_size) {
                tokens = tokens.slice(-meta.block_size);
            }

            const char = decode([nextToken]);
            updateDisplay(char);
            updateContextWindowDisplay(tokens);

            await new Promise(resolve => setTimeout(resolve, speed));

        } catch (error) {
            console.error('Generation error:', error);
            statusEl.textContent = 'ERROR: ' + error.message;
            stopGeneration();
            break;
        }
    }
}

function stopGeneration() {
    isGenerating = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'PAUSED — CLICK TEXT TO EDIT';
    statusEl.classList.remove('active');

    enableEditing();
}

function clearText() {
    stopGeneration();
    generatedText = '';
    cantoCount = 1;
    charsSinceLastCanto = 0;
    textOutput.textContent = '';
    textContainer.querySelectorAll('.canto-separator').forEach(el => el.remove());
    statusEl.textContent = 'CLEARED — PRESS START';
    disableEditing();
    resetDecoder(); // Clear pending UTF-8 bytes
}

// ============================================================================
// Initialization
// ============================================================================

async function initialize() {
    try {
        statusEl.textContent = 'LOADING MODEL...';
        statusEl.classList.add('loading');

        // Load metadata
        const metaResponse = await fetch('../../model/meta.json');
        if (!metaResponse.ok) {
            throw new Error('Failed to load meta.json');
        }
        meta = await metaResponse.json();
        initBPE(meta.merges);
        console.log('Loaded BPE tokenizer:', {
            vocab_size: meta.vocab_size,
            block_size: meta.block_size
        });

        // Load ONNX model
        statusEl.textContent = 'LOADING NEURAL NETWORK...';
        session = await ort.InferenceSession.create('../../model/model.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log('ONNX model loaded successfully');

        // Enable controls
        startBtn.disabled = false;
        statusEl.textContent = 'READY — PRESS START';
        statusEl.classList.remove('loading');

    } catch (error) {
        console.error('Initialization error:', error);
        statusEl.textContent = 'ERROR: ' + error.message;
        statusEl.classList.remove('loading');
    }
}

// ============================================================================
// Event Listeners Setup
// ============================================================================

function setupEventListeners() {
    // Get DOM elements
    textContainer = document.getElementById('text-container');
    textOutput = document.getElementById('text-output');
    cursorEl = document.getElementById('cursor');
    startBtn = document.getElementById('start-btn');
    stopBtn = document.getElementById('stop-btn');
    clearBtn = document.getElementById('clear-btn');
    temperatureSlider = document.getElementById('temperature');
    topKSlider = document.getElementById('top-k');
    topPSlider = document.getElementById('top-p');
    repPenaltySlider = document.getElementById('rep-penalty');
    speedSlider = document.getElementById('speed');
    tempValue = document.getElementById('temp-value');
    topKValue = document.getElementById('top-k-value');
    topPValue = document.getElementById('top-p-value');
    repPenaltyValue = document.getElementById('rep-penalty-value');
    speedValue = document.getElementById('speed-value');
    statusEl = document.getElementById('status');
    editIndicator = document.getElementById('edit-indicator');
    contextToggleBtn = document.getElementById('context-toggle-btn');
    contextWindowDisplay = document.getElementById('context-window-display');
    ctxContent = document.getElementById('ctx-content');
    ctxTokenCount = document.getElementById('ctx-token-count');

    // Control buttons
    startBtn.addEventListener('click', startGeneration);
    stopBtn.addEventListener('click', stopGeneration);
    clearBtn.addEventListener('click', clearText);

    // Context window toggle
    contextToggleBtn.addEventListener('click', () => {
        showContextWindow = !showContextWindow;
        contextToggleBtn.classList.toggle('active', showContextWindow);
        contextWindowDisplay.classList.toggle('visible', showContextWindow);
        contextToggleBtn.textContent = showContextWindow ? '⬛ hide context' : '⬚ show context';
    });

    // Temperature slider
    temperatureSlider.addEventListener('input', (e) => {
        temperature = parseFloat(e.target.value);
        tempValue.textContent = temperature.toFixed(2);
    });

    // Top-K slider
    topKSlider.addEventListener('input', (e) => {
        topK = parseInt(e.target.value);
        topKValue.textContent = topK;
    });

    // Top-P slider
    topPSlider.addEventListener('input', (e) => {
        topP = parseFloat(e.target.value);
        topPValue.textContent = topP.toFixed(2);
    });

    // Repetition Penalty slider
    repPenaltySlider.addEventListener('input', (e) => {
        repetitionPenalty = parseFloat(e.target.value);
        repPenaltyValue.textContent = repetitionPenalty.toFixed(2);
    });

    // Speed slider
    speedSlider.addEventListener('input', (e) => {
        speed = parseInt(e.target.value);
        speedValue.textContent = speed + 'ms';
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ignore if editing text
        if (textOutput.contentEditable === 'true' && document.activeElement === textOutput) {
            return;
        }

        // Space to toggle play/pause
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            if (isGenerating) {
                stopGeneration();
            } else {
                startGeneration();
            }
        }

        // Escape to stop
        if (e.code === 'Escape') {
            stopGeneration();
        }
    });

    // Touch gesture for mobile (tap to toggle)
    let lastTap = 0;
    textContainer.addEventListener('touchend', (e) => {
        if (isGenerating) return; // Don't handle taps while generating

        const currentTime = new Date().getTime();
        const tapLength = currentTime - lastTap;
        if (tapLength < 300 && tapLength > 0) {
            // Double tap - clear
            e.preventDefault();
            clearText();
        }
        lastTap = currentTime;
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initialize();
});
