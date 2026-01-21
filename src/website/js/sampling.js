/**
 * sampling.js
 * ===========
 * Advanced sampling functions for text generation.
 */

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
