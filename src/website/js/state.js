/**
 * state.js
 * ========
 * Global state variables, constants, and DOM element references.
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
const MOBILE_BLOCK_SIZE = 128; // Reduced context window for mobile
let effectiveBlockSize = 256;  // Will be set based on device

// Sampling Parameters (with defaults)
let temperature = 0.85;
let topK = 40;
let topP = 0.92;
let repetitionPenalty = 1.15;
let speed = 50;
let showContextWindow = false;
let currentTokens = [];

// Dante Rhyme Mode (terza rima: ABA BCB CDC...)
let danteRhymeMode = true; // Enabled by default
let verseEndings = [];      // Stores the ending sound of each verse
let currentVerseNumber = 0; // Current verse count (0-indexed)
let justForcedRhyme = false; // Track if we just forced a rhyming word

// DOM Elements (initialized after DOMContentLoaded)
let textContainer, textOutput, cursorEl;
let startBtn, stopBtn, clearBtn;
let temperatureSlider, topKSlider, topPSlider, repPenaltySlider, speedSlider;
let tempValue, topKValue, topPValue, repPenaltyValue, speedValue;
let statusEl, editIndicator;
let contextToggleBtn, contextWindowDisplay, ctxContent, ctxTokenCount;
