/**
 * tokenizer.js
 * ============
 * BPE Tokenizer functions for encoding and decoding text.
 */

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
