/**
 * main.js
 * =======
 * Initialization, device detection (WebGPU/WASM), model loading,
 * Service Worker registration, and extended user controls.
 */

// ============================================================================
// Initialization
// ============================================================================

async function initialize() {
    try {
        statusEl.textContent = 'LOADING MODEL...';
        statusEl.classList.add('loading');

        // Register Service Worker for offline capability & caching
        if ('serviceWorker' in navigator && window.location.protocol.startsWith('http')) {
            navigator.serviceWorker.register('./sw.js').then(() => {
                console.log('[SW] Service Worker registered successfully');
            }).catch(err => {
                console.log('[SW] Service Worker registration note:', err);
            });
        }

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

        // 1. Load original high-quality FP32 ONNX model (~47MB) with WASM provider
        const modelPath = '../../model/model.onnx';
        activeModelType = 'FP32';
        activeProvider = 'WASM';

        statusEl.textContent = `CARICAMENTO RETE NEURALE (WASM · ${activeModelType})...`;
        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log(`ONNX model loaded successfully: ${modelPath}`);

        // Enable controls
        startBtn.disabled = false;
        const benchmarkBtn = document.getElementById('benchmark-btn');
        if (benchmarkBtn) benchmarkBtn.disabled = false;
        statusEl.textContent = 'CARICAMENTO RIMARIO...';

        // Load rimario (non-blocking for core functionality)
        await loadRimario('../../model/rimario.json');

        statusEl.textContent = `PRONTO — PREMI AVVIA (${activeProvider} · ${activeModelType})`;
        statusEl.classList.remove('loading');

    } catch (error) {
        console.error('Initialization error:', error);
        statusEl.textContent = 'ERROR: ' + error.message;
        statusEl.classList.remove('loading');
    }
}

// ============================================================================
// Extended Feature Actions
// ============================================================================

function startWithCustomIncipit() {
    const input = document.getElementById('incipit-input');
    if (!input) return;
    const customVerse = input.value.trim();
    if (!customVerse) return;

    clearText();
    const ending = getEndingSound(customVerse);
    generatedText = `${customVerse}\n`;
    charsSinceLastCanto = generatedText.length;

    if (showMetricAnalysis) {
        renderFormattedText();
    } else {
        textOutput.textContent = generatedText;
    }

    verseEndings = [ending];
    currentVerseNumber = 1;
    resetRhymeSearchAttempts();
    input.value = '';

    console.log(`[INCIPIT] Started with custom verse: "${customVerse}", rhyme ending: "${ending}"`);
    startGeneration();
}

function toggleSpeech() {
    const speakBtn = document.getElementById('speak-btn');
    if (!('speechSynthesis' in window)) {
        alert('Sintesi vocale Web Speech API non supportata da questo browser.');
        return;
    }
    if (isSpeaking) {
        window.speechSynthesis.cancel();
        isSpeaking = false;
        if (speakBtn) {
            speakBtn.textContent = '🗣 recita';
            speakBtn.classList.remove('btn-speaking');
        }
        return;
    }

    const verses = generatedText.split('\n').map(v => v.trim()).filter(v => v.length > 0 && !v.startsWith('CANTO'));
    if (verses.length === 0) return;

    isSpeaking = true;
    if (speakBtn) {
        speakBtn.textContent = '⏹ ferma';
        speakBtn.classList.add('btn-speaking');
    }

    let verseIdx = 0;
    function speakNext() {
        if (!isSpeaking || verseIdx >= verses.length) {
            isSpeaking = false;
            if (speakBtn) {
                speakBtn.textContent = '🗣 recita';
                speakBtn.classList.remove('btn-speaking');
            }
            return;
        }

        const raw = verses[verseIdx].replace(/\[[A-Z]\]/g, '').trim();
        const utter = new SpeechSynthesisUtterance(raw);
        utter.lang = 'it-IT';
        utter.rate = 0.88;
        utter.pitch = 0.95;

        const voices = window.speechSynthesis.getVoices();
        const itVoice = voices.find(v => v.lang && (v.lang.startsWith('it') || v.name.toLowerCase().includes('italian')));
        if (itVoice) utter.voice = itVoice;

        utter.onend = () => {
            verseIdx++;
            setTimeout(speakNext, 350);
        };
        utter.onerror = () => {
            isSpeaking = false;
            if (speakBtn) {
                speakBtn.textContent = '🗣 recita';
                speakBtn.classList.remove('btn-speaking');
            }
        };

        window.speechSynthesis.speak(utter);
    }
    speakNext();
}

function toggleMetricAnalysis() {
    showMetricAnalysis = !showMetricAnalysis;
    const metricBtn = document.getElementById('metric-toggle-btn');
    if (metricBtn) {
        metricBtn.classList.toggle('active', showMetricAnalysis);
        metricBtn.textContent = showMetricAnalysis ? '📐 nascondi metrica' : '📐 metrica';
    }
    renderFormattedText();
}

function copyPoetry() {
    if (!generatedText.trim()) return;
    const lines = generatedText.split('\n').filter(l => l.trim().length > 0);
    const tercets = [];
    for (let i = 0; i < lines.length; i += 3) {
        tercets.push(lines.slice(i, i + 3).join('\n'));
    }
    const formatted = tercets.join('\n\n');
    navigator.clipboard.writeText(formatted).then(() => {
        const copyBtn = document.getElementById('copy-btn');
        if (copyBtn) {
            const old = copyBtn.textContent;
            copyBtn.textContent = '✓ Copiato!';
            setTimeout(() => { copyBtn.textContent = old; }, 2000);
        }
    });
}

function exportPoetry() {
    if (!generatedText.trim()) return;
    const blob = new Blob([generatedText], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `canto_dantesco_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
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

    // Extended Feature buttons
    const incipitBtn = document.getElementById('incipit-btn');
    const incipitInput = document.getElementById('incipit-input');
    if (incipitBtn) incipitBtn.addEventListener('click', startWithCustomIncipit);
    if (incipitInput) {
        incipitInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                startWithCustomIncipit();
            }
        });
    }

    const speakBtn = document.getElementById('speak-btn');
    if (speakBtn) speakBtn.addEventListener('click', toggleSpeech);

    const metricBtn = document.getElementById('metric-toggle-btn');
    if (metricBtn) metricBtn.addEventListener('click', toggleMetricAnalysis);

    const copyBtn = document.getElementById('copy-btn');
    if (copyBtn) copyBtn.addEventListener('click', copyPoetry);

    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) exportBtn.addEventListener('click', exportPoetry);
    
    // Benchmark button
    const benchmarkBtn = document.getElementById('benchmark-btn');
    const benchmarkClose = document.getElementById('benchmark-close');
    const benchmarkAgain = document.getElementById('benchmark-again');
    
    if (benchmarkBtn) benchmarkBtn.addEventListener('click', runBenchmark);
    if (benchmarkClose) benchmarkClose.addEventListener('click', closeBenchmarkModal);
    if (benchmarkAgain) {
        benchmarkAgain.addEventListener('click', () => {
            runBenchmark();
        });
    }

    // Context window toggle
    if (contextToggleBtn) {
        contextToggleBtn.addEventListener('click', () => {
            showContextWindow = !showContextWindow;
            contextToggleBtn.classList.toggle('active', showContextWindow);
            contextWindowDisplay.classList.toggle('visible', showContextWindow);
            contextToggleBtn.textContent = showContextWindow ? '\u2B1B hide context' : '\u2B1A show context';
        });
    }

    // Dante Rhyme Mode toggle
    const rhymeToggleBtn = document.getElementById('rhyme-toggle-btn');
    const rhymeStatus = document.getElementById('rhyme-status');
    const updateRhymeToggle = () => {
        if (!rhymeToggleBtn || !rhymeStatus) return;
        const active = danteRhymeMode !== RHYME_MODE_OFF;
        rhymeToggleBtn.classList.toggle('active', active);
        const forced = danteRhymeMode === RHYME_MODE_FORCED;
        const strict = danteRhymeMode === RHYME_MODE_STRICT;
        rhymeStatus.textContent = forced ? 'FORCED (RIMARIO)' :
            strict ? 'STRICT (BEST-EFFORT)' : danteRhymeMode;
        rhymeToggleBtn.textContent = forced ? '\uD83C\uDFAD terza rima FORCED (rimario)' :
            strict ? '\uD83C\uDFAD terza rima STRICT (best-effort)' :
            `\uD83C\uDFAD terza rima ${danteRhymeMode} (ABA BCB)`;
    };

    if (rhymeToggleBtn) {
        rhymeToggleBtn.addEventListener('click', () => {
            danteRhymeMode = danteRhymeMode === RHYME_MODE_FORCED ? RHYME_MODE_SOFT :
                danteRhymeMode === RHYME_MODE_SOFT ? RHYME_MODE_STRICT :
                danteRhymeMode === RHYME_MODE_STRICT ? RHYME_MODE_OFF : RHYME_MODE_FORCED;
            cancelPendingRhymeCompletion();
            updateRhymeToggle();
        });
        updateRhymeToggle();
    }

    // Sliders
    if (temperatureSlider && tempValue) {
        temperatureSlider.addEventListener('input', (e) => {
            temperature = parseFloat(e.target.value);
            tempValue.textContent = temperature.toFixed(2);
        });
    }

    if (topKSlider && topKValue) {
        topKSlider.addEventListener('input', (e) => {
            topK = parseInt(e.target.value);
            topKValue.textContent = topK;
        });
    }

    if (topPSlider && topPValue) {
        topPSlider.addEventListener('input', (e) => {
            topP = parseFloat(e.target.value);
            topPValue.textContent = topP.toFixed(2);
        });
    }

    if (repPenaltySlider && repPenaltyValue) {
        repPenaltySlider.addEventListener('input', (e) => {
            repetitionPenalty = parseFloat(e.target.value);
            repPenaltyValue.textContent = repetitionPenalty.toFixed(2);
        });
    }

    if (speedSlider && speedValue) {
        speedSlider.addEventListener('input', (e) => {
            speed = parseInt(e.target.value);
            speedValue.textContent = speed + 'ms';
        });
    }

    // Context Size slider
    const ctxSizeSlider = document.getElementById('ctx-size');
    const ctxSizeValue = document.getElementById('ctx-size-value');
    
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
        if (textOutput && textOutput.contentEditable === 'true' && document.activeElement === textOutput) {
            return;
        }
        if (document.activeElement && document.activeElement.tagName === 'INPUT') {
            return;
        }

        // Space to toggle play/pause
        if (e.code === 'Space' && (e.target === document.body || e.target === document.documentElement)) {
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
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initialize();
});
