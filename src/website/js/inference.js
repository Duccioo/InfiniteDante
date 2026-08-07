/**
 * inference.js
 * ============
 * ONNX model inference, ordinary sampling, and bounded rhyme completion search.
 *
 * Two rhyme-assisted modes:
 * - SOFT / STRICT:  beam search over model continuations, accept only those
 *   whose decoded text ends with a complete word matching the target suffix.
 * - FORCED:  pick a rhyme word from the rimario, encode its BPE sequence
 *   (space + word + newline), rank candidates by model probability + meter,
 *   with progressive fallbacks (exact+11, exact+10/12, assonance, forced).
 */

const RHYME_SEARCH_BEAM_WIDTH = 3;
const RHYME_SEARCH_CANDIDATE_POOL = 32;
const RHYME_SEARCH_MAX_TOKENS = 4;
const RHYME_SEARCH_MIN_CHARS = 28;
const RHYME_STRICT_MAX_ATTEMPTS = 3;
const RHYME_STRICT_RETRY_CHARS = 6;
const RHYME_EXACT_ENDING_BONUS = 4;

// FORCED mode constants
const FORCED_MAX_CANDIDATES = 30;
const FORCED_METER_TOLERANCE_TIGHT = 0;  // exactly 11
const FORCED_METER_TOLERANCE_LOOSE = 1;  // 10-12

/** Run inference on the ONNX model. */
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

/** Apply the same user-controlled sampling distribution used by normal output. */
async function getSamplingProbabilities(context) {
    let logits = await runInference(context);
    logits = applyRepetitionPenalty(logits, context.slice(-64), repetitionPenalty);
    logits = applyTopK(logits, topK);
    return applyTopP(softmax(logits, temperature), topP);
}

function getFiniteTopCandidates(probs, limit) {
    return Array.from(probs, (prob, tokenId) => ({ tokenId, prob }))
        .filter(candidate => Number.isFinite(candidate.prob) && candidate.prob > 0)
        .sort((first, second) => second.prob - first.prob || first.tokenId - second.tokenId)
        .slice(0, limit);
}

function decodeRhymeCandidate(tokenIds, pendingPrefix) {
    return decodeTokenByteSequences(tokenIds.map(tokenId => bpe_vocab[tokenId]), pendingPrefix);
}

function keepBestDistinctBeams(beams, limit) {
    const distinct = new Map();
    for (const beam of beams) {
        const key = beam.tokenIds.join(',');
        const existing = distinct.get(key);
        if (!existing || beam.rank > existing.rank) distinct.set(key, beam);
    }
    return Array.from(distinct.values())
        .sort((first, second) => second.rank - first.rank || first.tokenIds.length - second.tokenIds.length)
        .slice(0, limit);
}

function getSoftBeamHint(partialVerse, continuation, targetEnding) {
    if (danteRhymeMode !== RHYME_MODE_SOFT || continuation.includes('\n')) return 0;
    return getAssonanceScore(getEndingSound(`${partialVerse}${continuation}`), targetEnding) * 0.05;
}

function getBeamRank(logProbability, partialVerse, continuation, targetEnding) {
    if (continuation.includes('\n')) return logProbability;
    const progress = getExactEndingProgressScore(`${partialVerse}${continuation}`, targetEnding);
    return logProbability + progress * RHYME_EXACT_ENDING_BONUS +
        getSoftBeamHint(partialVerse, continuation, targetEnding);
}

/**
 * Search only viable model branches for a single completed, rhyming line.
 * At most nine extra ONNX calls are made per attempt (three beams across three
 * expansions). Each model distribution contributes a CPU-only pool of 32 tokens.
 */
async function findRhymeCompletion(context, initialProbs, partialVerse, targetEnding, pendingPrefix,
    isSearchCurrent = () => true, getBranchProbabilities = getSamplingProbabilities) {
    if (!isSearchCurrent()) return { canceled: true };
    let beams = getFiniteTopCandidates(initialProbs, RHYME_SEARCH_CANDIDATE_POOL).map(candidate => {
        const tokenIds = [candidate.tokenId];
        const text = decodeRhymeCandidate(tokenIds, pendingPrefix);
        return {
            tokenIds,
            text,
            logProbability: Math.log(candidate.prob),
            rank: getBeamRank(Math.log(candidate.prob), partialVerse, text, targetEnding)
        };
    });

    for (let depth = 1; depth <= RHYME_SEARCH_MAX_TOKENS && beams.length; depth++) {
        if (!isSearchCurrent()) return { canceled: true };
        const completed = [];
        const expandable = [];
        for (const beam of beams) {
            const validation = evaluateCompletedRhymeContinuation(partialVerse, beam.text, targetEnding);
            if (validation.accepted) {
                completed.push({ ...beam, validation, rank: beam.rank + validation.meterScore * 0.1 });
            } else if (validation.reason === 'no-newline') {
                expandable.push(beam);
            }
        }
        if (completed.length) {
            return completed.sort((first, second) => second.rank - first.rank)[0];
        }
        if (depth === RHYME_SEARCH_MAX_TOKENS) break;

        const expanded = [];
        for (const beam of keepBestDistinctBeams(expandable, RHYME_SEARCH_BEAM_WIDTH)) {
            if (!isSearchCurrent()) return { canceled: true };
            const branchContext = context.concat(beam.tokenIds).slice(-effectiveBlockSize);
            const branchProbs = await getBranchProbabilities(branchContext);
            if (!isSearchCurrent()) return { canceled: true };
            for (const candidate of getFiniteTopCandidates(branchProbs, RHYME_SEARCH_CANDIDATE_POOL)) {
                const tokenIds = beam.tokenIds.concat(candidate.tokenId);
                const text = decodeRhymeCandidate(tokenIds, pendingPrefix);
                if ((text.match(/\n/g) || []).length > 1) continue;
                const logProbability = beam.logProbability + Math.log(candidate.prob);
                expanded.push({
                    tokenIds,
                    text,
                    logProbability,
                    rank: getBeamRank(logProbability, partialVerse, text, targetEnding)
                });
            }
        }
        beams = keepBestDistinctBeams(expanded, RHYME_SEARCH_BEAM_WIDTH);
    }
    return null;
}

// =========================================================================
// FORCED mode: select a word from the rimario and force the BPE ending
// =========================================================================

/**
 * Encode a word ending as BPE tokens: " " + word + "\n"
 * Returns an array of token IDs.
 */
function encodeWordEnding(word) {
    return encode(' ' + word + '\n');
}

/**
 * Compute the model's log-probability for a specific token sequence
 * given a context, WITHOUT running inference per-token (we do one call
 * for the whole continuation using teacher-forced logits).
 *
 * Since our ONNX model outputs logits for all positions, we can score
 * the full sequence in one inference call by appending the tokens and
 * reading off the logits at each position.
 */
async function scoreTokenSequence(context, tokenIds) {
    const fullSeq = context.concat(tokenIds).slice(-effectiveBlockSize);
    const seqLen = fullSeq.length;
    const inputArray = new BigInt64Array(fullSeq.map(t => BigInt(t)));
    const inputTensor = new ort.Tensor('int64', inputArray, [1, seqLen]);
    const results = await session.run({ input: inputTensor });
    const output = results.output;
    const vocabSize = meta.vocab_size;

    // Sum log-probs for each forced token
    let logProb = 0;
    const startPos = seqLen - tokenIds.length;
    for (let i = 0; i < tokenIds.length; i++) {
        const pos = startPos + i;
        const logitsStart = pos * vocabSize;
        // Compute log-softmax for this position
        const logits = output.data.slice(logitsStart, logitsStart + vocabSize);
        let maxLogit = -Infinity;
        for (let j = 0; j < vocabSize; j++) {
            if (logits[j] > maxLogit) maxLogit = logits[j];
        }
        let sumExp = 0;
        for (let j = 0; j < vocabSize; j++) {
            sumExp += Math.exp(logits[j] - maxLogit);
        }
        const tokenLogProb = (logits[tokenIds[i]] - maxLogit) - Math.log(sumExp);
        logProb += tokenLogProb;
    }

    return logProb;
}

/**
 * FORCED rhyme completion: pick a rhyme word from the rimario,
 * encode it as BPE, score with the model, apply progressive fallbacks.
 *
 * Fallback levels:
 * 1. Exact rhyme + 11 syllables
 * 2. Exact rhyme + 10-12 syllables
 * 3. Assonance + 11 syllables
 * 4. Forced insertion of best word (ignore meter)
 * 5. Simple close (normal sampling)
 */
async function findForcedRhymeCompletion(context, partialVerse, targetSuffix,
    isSearchCurrent = () => true) {
    if (!rimario || !targetSuffix) return null;

    // Get candidate rhyme words from rimario
    const exactWords = getRhymeFamilies(targetSuffix);
    if (!exactWords.length) return null;

    // Filter out already-used rhyme words
    const availableExact = exactWords.filter(w => !usedRhymeWords.has(w));
    if (!availableExact.length && exactWords.length > 0) {
        // All words used; allow reuse but still score them
        availableExact.push(...exactWords);
    }

    // Also collect assonance candidates (same vowels in suffix, different consonants)
    const targetVowels = targetSuffix.replace(/[^aeiou]/g, '');
    const assonanceCandidates = [];
    if (rimario && targetVowels.length >= 2) {
        for (const [suffix, words] of Object.entries(rimario)) {
            if (suffix === targetSuffix) continue;
            const suffixVowels = suffix.replace(/[^aeiou]/g, '');
            if (suffixVowels === targetVowels) {
                for (const w of words) {
                    if (!usedRhymeWords.has(w)) {
                        assonanceCandidates.push({ word: w, suffix, type: 'assonance' });
                    }
                }
            }
        }
    }

    // Score exact candidates
    const scored = [];
    const limit = Math.min(availableExact.length, FORCED_MAX_CANDIDATES);

    for (let i = 0; i < limit; i++) {
        if (!isSearchCurrent()) return { canceled: true };

        const word = availableExact[i];
        const tokenIds = encodeWordEnding(word);
        const fullLine = partialVerse + ' ' + word;
        const syllables = countItalianSyllables(fullLine);
        const meterDev = Math.abs(syllables - 11);
        const logProb = await scoreTokenSequence(context, tokenIds);

        scored.push({
            word,
            tokenIds,
            logProb,
            syllables,
            meterDev,
            type: 'exact',
            line: fullLine
        });
    }

    // Score assonance candidates (fewer, only if exact fails)
    const assonanceScored = [];
    const assonanceLimit = Math.min(assonanceCandidates.length, 10);
    for (let i = 0; i < assonanceLimit; i++) {
        if (!isSearchCurrent()) return { canceled: true };

        const { word } = assonanceCandidates[i];
        const tokenIds = encodeWordEnding(word);
        const fullLine = partialVerse + ' ' + word;
        const syllables = countItalianSyllables(fullLine);
        const meterDev = Math.abs(syllables - 11);
        const logProb = await scoreTokenSequence(context, tokenIds);

        assonanceScored.push({
            word,
            tokenIds,
            logProb,
            syllables,
            meterDev,
            type: 'assonance',
            line: fullLine
        });
    }

    // Progressive fallback selection
    const rankByProb = (a, b) => b.logProb - a.logProb;

    // Level 1: exact rhyme + exactly 11 syllables
    let best = scored.filter(c => c.meterDev === FORCED_METER_TOLERANCE_TIGHT)
        .sort(rankByProb)[0];
    if (best) {
        return buildForcedResult(best, 'exact+11');
    }

    // Level 2: exact rhyme + 10-12 syllables
    best = scored.filter(c => c.meterDev <= FORCED_METER_TOLERANCE_LOOSE)
        .sort(rankByProb)[0];
    if (best) {
        return buildForcedResult(best, 'exact+10-12');
    }

    // Level 3: assonance + 11 syllables
    best = assonanceScored.filter(c => c.meterDev === FORCED_METER_TOLERANCE_TIGHT)
        .sort(rankByProb)[0];
    if (best) {
        return buildForcedResult(best, 'assonance+11');
    }

    // Level 4: forced insertion of best exact word (ignore meter)
    best = scored.sort(rankByProb)[0];
    if (best) {
        return buildForcedResult(best, 'forced-insert');
    }

    // Level 5: nothing found
    return null;
}

function buildForcedResult(candidate, level) {
    const ending = getEndingSound(candidate.line);
    return {
        tokenIds: candidate.tokenIds,
        text: ' ' + candidate.word + '\n',
        logProbability: candidate.logProb,
        rank: candidate.logProb,
        validation: {
            accepted: true,
            line: candidate.line,
            ending,
            meterScore: getVerseMeterScore(candidate.line),
            forcedLevel: level,
            word: candidate.word
        }
    };
}

/** Generate the next token with advanced sampling. */
async function generateNext(context) {
    if (pendingRhymeTokenQueue.length) return pendingRhymeTokenQueue.shift();

    const invocationSnapshot = Object.freeze({
        epoch: rhymeGenerationEpoch,
        mode: danteRhymeMode,
        verseNumber: currentVerseNumber,
        contextKey: context.join(','),
        generatedText
    });
    const probs = await getSamplingProbabilities(context);
    const afterInitialInference = {
        epoch: rhymeGenerationEpoch,
        mode: danteRhymeMode,
        verseNumber: currentVerseNumber,
        contextKey: currentTokens.join(','),
        generatedText
    };
    if (!isRhymeSearchSnapshotCurrent(invocationSnapshot, afterInitialInference)) {
        return fallbackToNormalSampling(probs, sample);
    }
    if (danteRhymeMode === RHYME_MODE_OFF) return sample(probs);

    const rhymeTarget = getRhymeTarget(currentVerseNumber);
    const partialVerse = getCurrentPartialVerse();
    const targetEnding = rhymeTarget >= 0 && rhymeTarget < verseEndings.length ?
        verseEndings[rhymeTarget] : '';

    // FORCED mode: use rimario-based completion
    if (danteRhymeMode === RHYME_MODE_FORCED && targetEnding && partialVerse.length >= RHYME_SEARCH_MIN_CHARS) {
        const isSearchCurrent = () => isRhymeSearchSnapshotCurrent(invocationSnapshot, {
            epoch: rhymeGenerationEpoch,
            mode: danteRhymeMode,
            verseNumber: currentVerseNumber,
            contextKey: currentTokens.join(','),
            generatedText
        });

        const completion = await findForcedRhymeCompletion(
            context, partialVerse, targetEnding, isSearchCurrent
        );

        const currentSnapshot = {
            epoch: rhymeGenerationEpoch,
            mode: danteRhymeMode,
            verseNumber: currentVerseNumber,
            contextKey: currentTokens.join(','),
            generatedText
        };
        if (!isRhymeSearchSnapshotCurrent(invocationSnapshot, currentSnapshot)) {
            return fallbackToNormalSampling(probs, sample);
        }
        if (completion && completion.canceled) return fallbackToNormalSampling(probs, sample);
        if (completion) {
            pendingRhymeTokenQueue = completion.tokenIds.slice();
            usedRhymeWords.add(completion.validation.word);
            console.log(`[RHYME] FORCED selected "${completion.validation.word}" (${completion.validation.forcedLevel}) for target "${targetEnding}"`);
            return pendingRhymeTokenQueue.shift();
        }
        // No forced completion found; fall through to normal sampling
        console.warn(`[RHYME] FORCED found no rhyme word for "${targetEnding}"; using normal sampling.`);
        return sample(probs);
    }

    // SOFT / STRICT mode: beam search
    const maxAttempts = danteRhymeMode === RHYME_MODE_STRICT ? RHYME_STRICT_MAX_ATTEMPTS : 1;
    const retryDistanceMet = danteRhymeMode !== RHYME_MODE_STRICT ||
        partialVerse.length - rhymeSearchLastAttemptLength >= RHYME_STRICT_RETRY_CHARS;
    const canSearch = targetEnding && partialVerse.length >= RHYME_SEARCH_MIN_CHARS &&
        rhymeSearchAttemptCount < maxAttempts &&
        retryDistanceMet;

    if (!canSearch) return sample(probs);

    rhymeSearchAttemptCount++;
    rhymeSearchLastAttemptLength = partialVerse.length;
    const pendingPrefix = typeof pendingBytes === 'undefined' ? new Uint8Array(0) : pendingBytes.slice();
    const isSearchCurrent = () => isRhymeSearchSnapshotCurrent(invocationSnapshot, {
        epoch: rhymeGenerationEpoch,
        mode: danteRhymeMode,
        verseNumber: currentVerseNumber,
        contextKey: currentTokens.join(','),
        generatedText
    });
    const completion = await findRhymeCompletion(
        context, probs, partialVerse, targetEnding, pendingPrefix, isSearchCurrent
    );
    const currentSnapshot = {
        epoch: rhymeGenerationEpoch,
        mode: danteRhymeMode,
        verseNumber: currentVerseNumber,
        contextKey: currentTokens.join(','),
        generatedText
    };
    if (!isRhymeSearchSnapshotCurrent(invocationSnapshot, currentSnapshot)) {
        return fallbackToNormalSampling(probs, sample);
    }
    if (completion && completion.canceled) return fallbackToNormalSampling(probs, sample);
    if (completion) {
        pendingRhymeTokenQueue = completion.tokenIds.slice();
        // Track the rhyme word if we can identify it
        if (completion.validation && completion.validation.ending) {
            const lineText = completion.validation.line || '';
            const lastWordMatch = lineText.match(/[a-zàèéìíòóùú]+$/i);
            if (lastWordMatch) usedRhymeWords.add(lastWordMatch[0].toLowerCase());
        }
        console.log(`[RHYME] ${danteRhymeMode} selected "${completion.validation.ending}" for target "${targetEnding}"`);
        return pendingRhymeTokenQueue.shift();
    }

    if (danteRhymeMode === RHYME_MODE_STRICT) {
        console.warn(`[RHYME] STRICT search found no completed rhyme for "${targetEnding}"; using normal sampling.`);
        return fallbackToNormalSampling(probs, sample);
    }
    return sample(probs);
}
