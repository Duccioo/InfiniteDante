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

    // Dante Rhyme Mode: boost tokens that contribute to correct rhymes
    if (danteRhymeMode && isNearVerseEnd()) {
        const rhymeTarget = getRhymeTarget(currentVerseNumber);
        
        // Debug logging
        console.log(`[RHYME] currentVerse=${currentVerseNumber}, rhymeTarget=${rhymeTarget}, verseEndings.length=${verseEndings.length}`);
        
        if (rhymeTarget >= 0 && rhymeTarget < verseEndings.length) {
            const targetEnding = verseEndings[rhymeTarget];
            console.log(`[RHYME] Target ending: "${targetEnding}"`);
            
            if (targetEnding) {
                // Get top candidates
                const indexed = probs.map((p, i) => ({ prob: p, idx: i }));
                indexed.sort((a, b) => b.prob - a.prob);
                
                // Check top candidates for rhyming potential
                const topCandidates = indexed.slice(0, Math.min(topK * 3, 150));
                
                let foundRhymingToken = false;
                let bestRhymeIdx = -1;
                let bestRhymeProb = 0;
                let boostCount = 0;
                
                for (const candidate of topCandidates) {
                    const rhymeScore = calculateRhymeScoreForToken(candidate.idx, targetEnding);
                    
                    if (rhymeScore >= 1.0) {
                        // Token ends verse with perfect rhyme - boost significantly
                        probs[candidate.idx] *= 5.0;
                        foundRhymingToken = true;
                        boostCount++;
                    } else if (rhymeScore >= 0.5) {
                        // Token contributes to potential rhyme - moderate boost
                        probs[candidate.idx] *= 2.0;
                        boostCount++;
                        if (candidate.prob > bestRhymeProb) {
                            bestRhymeProb = candidate.prob;
                            bestRhymeIdx = candidate.idx;
                        }
                    }
                }
                
                console.log(`[RHYME] Boosted ${boostCount} tokens, found perfect rhyme: ${foundRhymingToken}`);
                
                // Re-normalize probabilities
                const sum = probs.reduce((a, b) => a + b, 0);
                if (sum > 0) {
                    probs = probs.map(p => p / sum);
                }
            }
        }
    }

    return sample(probs);
}
