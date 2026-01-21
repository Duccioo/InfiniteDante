/**
 * main.js
 * =======
 * Initialization and event listeners setup.
 */

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
        
        // Detect mobile and set effective block size
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) 
                        || window.innerWidth < 768;
        effectiveBlockSize = isMobile ? MOBILE_BLOCK_SIZE : meta.block_size;
        
        console.log('Loaded BPE tokenizer:', {
            vocab_size: meta.vocab_size,
            block_size: meta.block_size,
            effective_block_size: effectiveBlockSize,
            is_mobile: isMobile
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
        document.getElementById('benchmark-btn').disabled = false;
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
    
    // Benchmark button
    const benchmarkBtn = document.getElementById('benchmark-btn');
    const benchmarkClose = document.getElementById('benchmark-close');
    const benchmarkAgain = document.getElementById('benchmark-again');
    
    benchmarkBtn.addEventListener('click', runBenchmark);
    benchmarkClose.addEventListener('click', closeBenchmarkModal);
    benchmarkAgain.addEventListener('click', () => {
        runBenchmark();
    });

    // Context window toggle
    contextToggleBtn.addEventListener('click', () => {
        showContextWindow = !showContextWindow;
        contextToggleBtn.classList.toggle('active', showContextWindow);
        contextWindowDisplay.classList.toggle('visible', showContextWindow);
        contextToggleBtn.textContent = showContextWindow ? '⬛ hide context' : '⬚ show context';
    });

    // Dante Rhyme Mode toggle
    const rhymeToggleBtn = document.getElementById('rhyme-toggle-btn');
    const rhymeStatus = document.getElementById('rhyme-status');
    
    rhymeToggleBtn.addEventListener('click', () => {
        danteRhymeMode = !danteRhymeMode;
        rhymeToggleBtn.classList.toggle('active', danteRhymeMode);
        rhymeStatus.textContent = danteRhymeMode ? 'ON' : 'OFF';
        rhymeToggleBtn.textContent = danteRhymeMode ? '🎭 terza rima (ABA BCB)' : '🎭 terza rima OFF';
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

    // Context Size slider
    const ctxSizeSlider = document.getElementById('ctx-size');
    const ctxSizeValue = document.getElementById('ctx-size-value');
    
    // Initialize slider with current effective block size
    if (ctxSizeSlider && ctxSizeValue) {
        ctxSizeSlider.value = effectiveBlockSize;
        ctxSizeValue.textContent = effectiveBlockSize;
        
        ctxSizeSlider.addEventListener('input', (e) => {
            effectiveBlockSize = parseInt(e.target.value);
            ctxSizeValue.textContent = effectiveBlockSize;
        });
    }

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

    // Touch gesture for mobile - disabled double-tap clear to prevent accidental text deletion
    // Users can use the clear button instead
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initialize();
});
