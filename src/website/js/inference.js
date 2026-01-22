/**
 * inference.js
 * ============
 * ONNX model inference and token generation.
 */

// ============================================================================
// Model Inference
// ============================================================================

/**
 * Run inference on the ONNX model.
 */
async function runInference(tokens) {
    const inputTokens = tokens.slice(-effectiveBlockSize);
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

    // Dante Rhyme Mode: force rhyming endings when needed
    if (danteRhymeMode) {
        // STAGE 3: If we just forced a rhyming word, now force a newline/punctuation to end the verse
        if (justForcedRhyme) {
            justForcedRhyme = false;
            console.log(`[RHYME] Completing verse after forced rhyme`);
            
            // Look for tokens that are just newline or punctuation+newline
            // Simple heuristic updates: find tokens containing newlines
            // Boost them massively to ensure line break
            for (let id = 0; id < probs.length; id++) {
                if (bpe_vocab[id]) {
                    const bytes = bpe_vocab[id];
                    const text = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
                    if (text.includes('\n')) {
                        probs[id] *= 1000.0;
                    }
                }
            }
        } 
        else {
            const rhymeTarget = getRhymeTarget(currentVerseNumber);
            const partialVerse = getCurrentPartialVerse();
            const verseLength = partialVerse.length;
            
            // Only examine rhymes when we have a target and verse is getting substantial
            if (rhymeTarget >= 0 && rhymeTarget < verseEndings.length && verseLength >= 25) {
                const targetEnding = verseEndings[rhymeTarget];
                
                if (targetEnding) {
                    // Find ALL tokens in vocabulary that would create a rhyme (word ending)
                    const rhymingTokens = findRhymingTokens(targetEnding, probs);
                    
                    if (rhymingTokens.length > 0) {
                        // STAGE 2: Force rhyme selection if verse is long
                        const shouldForce = verseLength >= 45;
                        
                        if (shouldForce) {
                            console.log(`[RHYME] FORCING rhyme for "${targetEnding}" at len ${verseLength}`);
                             // Select a rhyming token
                            const totalProb = rhymingTokens.reduce((sum, t) => sum + t.prob + 0.0001, 0);
                            let r = Math.random() * totalProb;
                            
                            for (const token of rhymingTokens) {
                                r -= (token.prob + 0.0001);
                                if (r <= 0) {
                                    console.log(`[RHYME] Selected forced rhyme: "${token.text}"`);
                                    justForcedRhyme = true;
                                    return token.tokenId;
                                }
                            }
                            // Fallback
                            justForcedRhyme = true;
                            return rhymingTokens[0].tokenId;
                        }
                        
                        // STAGE 1: Boost rhyming words (verse length 25-45)
                        // console.log(`[RHYME] Boosting ${rhymingTokens.length} rhymes for "${targetEnding}"`);
                        for (const token of rhymingTokens.slice(0, 50)) {
                             // Strong boost
                            probs[token.tokenId] *= 20.0;
                        }
                    } 
                }
            }
        }
        
        // Re-normalize after any boosting
        const sum = probs.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            probs = probs.map(p => p / sum);
        }
    }

    return sample(probs);
}
