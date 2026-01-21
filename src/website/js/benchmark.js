/**
 * benchmark.js
 * ============
 * Device performance benchmark functionality.
 */

// ============================================================================
// Benchmark Mode
// ============================================================================

let isBenchmarking = false;
const BENCHMARK_CONTEXT_SIZE = 256;
const BENCHMARK_TARGET_VERSES = 9; // 3 terzine = 9 verses

/**
 * Run a device performance benchmark by generating 3 terzine (9 verses).
 */
async function runBenchmark() {
    if (isBenchmarking || isGenerating) return;
    
    isBenchmarking = true;
    const benchmarkBtn = document.getElementById('benchmark-btn');
    const benchmarkModal = document.getElementById('benchmark-modal');
    const benchmarkBody = document.getElementById('benchmark-body');
    const benchmarkOutput = document.getElementById('benchmark-output');
    
    // Disable buttons during benchmark
    benchmarkBtn.disabled = true;
    startBtn.disabled = true;
    stopBtn.disabled = true;
    
    // Show modal with progress
    benchmarkBody.innerHTML = `
        <div class="benchmark-progress">
            <div class="benchmark-progress-text">INITIALIZING BENCHMARK...</div>
            <div class="benchmark-progress-bar">
                <div class="benchmark-progress-fill" id="benchmark-progress-fill" style="width: 0%"></div>
            </div>
        </div>
    `;
    benchmarkModal.classList.add('visible');
    
    // Benchmark state
    const savedBlockSize = effectiveBlockSize;
    effectiveBlockSize = BENCHMARK_CONTEXT_SIZE;
    
    const startPrompt = 'Nel mezzo del cammin di nostra vita ';
    let benchmarkText = startPrompt;
    let tokens = encode(benchmarkText);
    let versesGenerated = 0;
    let tokensGenerated = 0;
    const tokenTimes = [];
    
    const startTime = performance.now();
    
    try {
        // Generate until we have 9 verses (3 terzine)
        while (versesGenerated < BENCHMARK_TARGET_VERSES && isBenchmarking) {
            const tokenStart = performance.now();
            
            const nextToken = await generateNextBenchmark(tokens);
            tokens.push(nextToken);
            tokensGenerated++;
            
            if (tokens.length > BENCHMARK_CONTEXT_SIZE) {
                tokens = tokens.slice(-BENCHMARK_CONTEXT_SIZE);
            }
            
            const char = decodeBenchmark([nextToken]);
            benchmarkText += char;
            
            // Count newlines as verse endings
            if (char.includes('\n')) {
                versesGenerated++;
            }
            
            const tokenEnd = performance.now();
            tokenTimes.push(tokenEnd - tokenStart);
            
            // Update progress
            const progress = Math.min((versesGenerated / BENCHMARK_TARGET_VERSES) * 100, 100);
            const progressFill = document.getElementById('benchmark-progress-fill');
            const progressText = benchmarkBody.querySelector('.benchmark-progress-text');
            
            if (progressFill) {
                progressFill.style.width = `${progress}%`;
            }
            if (progressText) {
                progressText.textContent = `GENERATING VERSE ${versesGenerated + 1} / ${BENCHMARK_TARGET_VERSES}...`;
            }
        }
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        
        // Calculate statistics
        const tokensPerSecond = (tokensGenerated / totalTime) * 1000;
        const avgTimePerToken = totalTime / tokensGenerated;
        
        // Restore block size
        effectiveBlockSize = savedBlockSize;
        
        // Display results
        showBenchmarkResults({
            text: benchmarkText,
            totalTime: totalTime,
            tokensGenerated: tokensGenerated,
            tokensPerSecond: tokensPerSecond,
            avgTimePerToken: avgTimePerToken,
            versesGenerated: versesGenerated,
            contextWindow: BENCHMARK_CONTEXT_SIZE
        });
        
    } catch (error) {
        console.error('Benchmark error:', error);
        effectiveBlockSize = savedBlockSize;
        benchmarkBody.innerHTML = `
            <div class="benchmark-progress" style="color: #e74c3c;">
                <div class="benchmark-progress-text">ERROR: ${error.message}</div>
            </div>
        `;
    }
    
    isBenchmarking = false;
    benchmarkBtn.disabled = false;
    startBtn.disabled = false;
}

/**
 * Generate next token for benchmark (no delay, pure performance).
 */
async function generateNextBenchmark(context) {
    let logits = await runInference(context);
    
    // Apply repetition penalty
    const recentTokens = context.slice(-64);
    logits = applyRepetitionPenalty(logits, recentTokens, repetitionPenalty);
    
    // Apply Top-K
    logits = applyTopK(logits, topK);
    
    // Convert to probabilities
    let probs = softmax(logits, temperature);
    
    // Apply Top-P
    probs = applyTopP(probs, topP);
    
    return sample(probs);
}

/**
 * Decode tokens for benchmark (separate buffer from main decoder).
 */
let benchmarkPendingBytes = new Uint8Array(0);

function decodeBenchmark(tokens) {
    const tokenLength = tokens.reduce((acc, t) => acc + (bpe_vocab[t] ? bpe_vocab[t].length : 0), 0);
    const result = new Uint8Array(benchmarkPendingBytes.length + tokenLength);
    
    result.set(benchmarkPendingBytes, 0);
    let offset = benchmarkPendingBytes.length;
    
    for (const t of tokens) {
        if (bpe_vocab[t]) {
            result.set(bpe_vocab[t], offset);
            offset += bpe_vocab[t].length;
        }
    }
    
    // Find valid UTF-8 boundary
    let validEnd = result.length;
    for (let i = Math.max(0, result.length - 4); i < result.length; i++) {
        const byte = result[i];
        if ((byte & 0x80) === 0) {
            validEnd = i + 1;
        } else if ((byte & 0xE0) === 0xC0) {
            if (i + 2 <= result.length) validEnd = i + 2;
            else { validEnd = i; break; }
        } else if ((byte & 0xF0) === 0xE0) {
            if (i + 3 <= result.length) validEnd = i + 3;
            else { validEnd = i; break; }
        } else if ((byte & 0xF8) === 0xF0) {
            if (i + 4 <= result.length) validEnd = i + 4;
            else { validEnd = i; break; }
        }
    }
    
    if (validEnd < result.length) {
        benchmarkPendingBytes = result.slice(validEnd);
    } else {
        benchmarkPendingBytes = new Uint8Array(0);
    }
    
    return new TextDecoder('utf-8', { fatal: false }).decode(result.slice(0, validEnd));
}

/**
 * Display benchmark results in the modal.
 */
function showBenchmarkResults(stats) {
    const benchmarkBody = document.getElementById('benchmark-body');
    const benchmarkOutput = document.getElementById('benchmark-output');
    
    // Reset benchmark pending bytes
    benchmarkPendingBytes = new Uint8Array(0);
    
    benchmarkBody.innerHTML = `
        <div class="benchmark-output" id="benchmark-output"></div>
        <div class="benchmark-stats" id="benchmark-stats">
            <div class="stat-row">
                <span class="stat-label">Total Time</span>
                <span class="stat-value" id="stat-total-time">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Tokens Generated</span>
                <span class="stat-value" id="stat-tokens">-</span>
            </div>
            <div class="stat-row highlight">
                <span class="stat-label">Tokens/Second</span>
                <span class="stat-value" id="stat-tps">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Time/Token</span>
                <span class="stat-value" id="stat-avg-time">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Context Window</span>
                <span class="stat-value" id="stat-ctx">256</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Verses Generated</span>
                <span class="stat-value" id="stat-verses">-</span>
            </div>
        </div>
    `;
    
    // Update values
    document.getElementById('benchmark-output').textContent = stats.text;
    document.getElementById('stat-total-time').textContent = formatTime(stats.totalTime);
    document.getElementById('stat-tokens').textContent = stats.tokensGenerated.toString();
    document.getElementById('stat-tps').textContent = stats.tokensPerSecond.toFixed(2) + ' tok/s';
    document.getElementById('stat-avg-time').textContent = stats.avgTimePerToken.toFixed(2) + ' ms';
    document.getElementById('stat-ctx').textContent = stats.contextWindow.toString();
    document.getElementById('stat-verses').textContent = `${stats.versesGenerated} (3 terzine)`;
}

/**
 * Format milliseconds to readable time string.
 */
function formatTime(ms) {
    if (ms < 1000) {
        return ms.toFixed(0) + ' ms';
    } else if (ms < 60000) {
        return (ms / 1000).toFixed(2) + ' s';
    } else {
        const minutes = Math.floor(ms / 60000);
        const seconds = ((ms % 60000) / 1000).toFixed(1);
        return `${minutes}m ${seconds}s`;
    }
}

/**
 * Close benchmark modal.
 */
function closeBenchmarkModal() {
    const benchmarkModal = document.getElementById('benchmark-modal');
    benchmarkModal.classList.remove('visible');
    isBenchmarking = false;
}
