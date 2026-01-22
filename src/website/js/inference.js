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
        const rhymeTarget = getRhymeTarget(currentVerseNumber);
        const partialVerse = getCurrentPartialVerse();
        const verseLength = partialVerse.length;
        
        // Only log when we have a rhyme target
        if (rhymeTarget >= 0 && rhymeTarget < verseEndings.length) {
            const targetEnding = verseEndings[rhymeTarget];
            
            // When verse is getting long, try to force a rhyming ending
            if (verseLength >= 35 && targetEnding) {
                console.log(`[RHYME] Verse ${currentVerseNumber}, length=${verseLength}, target="${targetEnding}"`);
                
                // Find ALL tokens in vocabulary that would create a rhyme
                const rhymingTokens = findRhymingNewlineTokens(targetEnding, probs);
                
                if (rhymingTokens.length > 0) {
                    console.log(`[RHYME] Found ${rhymingTokens.length} rhyming tokens, best: "${rhymingTokens[0].text}"`);
                    
                    // Create a new probability distribution that strongly favors rhyming tokens
                    // The longer the verse, the more we force the rhyme
                    const forceStrength = Math.min((verseLength - 35) / 20, 1.0); // 0 at 35, 1 at 55+
                    
                    if (Math.random() < 0.3 + forceStrength * 0.5) {
                        // Force selection from rhyming tokens
                        // Weight by original probability
                        const totalProb = rhymingTokens.reduce((sum, t) => sum + t.prob + 0.0001, 0);
                        let r = Math.random() * totalProb;
                        
                        for (const token of rhymingTokens) {
                            r -= (token.prob + 0.0001);
                            if (r <= 0) {
                                console.log(`[RHYME] FORCED rhyme: "${token.text}" (ending: ${token.ending})`);
                                return token.tokenId;
                            }
                        }
                        // Fallback to first rhyming token
                        console.log(`[RHYME] FORCED first rhyme: "${rhymingTokens[0].text}"`);
                        return rhymingTokens[0].tokenId;
                    }
                } else {
                    console.log(`[RHYME] No rhyming tokens found for "${targetEnding}"`);
                }
                
                // Even if not forcing, still boost rhyming tokens
                for (const token of rhymingTokens.slice(0, 50)) {
                    probs[token.tokenId] *= 10.0;
                }
                
                // Re-normalize
                const sum = probs.reduce((a, b) => a + b, 0);
                if (sum > 0) {
                    probs = probs.map(p => p / sum);
                }
            }
        }
    }

    return sample(probs);
}
