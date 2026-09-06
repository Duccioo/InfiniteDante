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
const RHYME_SEARCH_MAX_TOKENS = 8;
const RHYME_SEARCH_MIN_CHARS = 26;
const RHYME_STRICT_MAX_ATTEMPTS = 3;
const RHYME_STRICT_RETRY_CHARS = 6;
const RHYME_EXACT_ENDING_BONUS = 4;

// FORCED mode constants
const FORCED_MAX_CANDIDATES = 25;
const FORCED_METER_TOLERANCE_TIGHT = 0;  // exactly 11
const FORCED_METER_TOLERANCE_LOOSE = 1;  // 10-12

let newlineTokenIds = null;
function getNewlineTokenIds() {
    if (newlineTokenIds) return newlineTokenIds;
    newlineTokenIds = new Set();
    if (typeof meta !== 'undefined' && meta && typeof bpe_vocab !== 'undefined') {
        for (let id = 0; id < meta.vocab_size; id++) {
            const bytes = bpe_vocab[id];
            if (bytes && bytes.includes(10)) { // 10 is '\n'
                newlineTokenIds.add(id);
            }
        }
    } else {
        newlineTokenIds.add(10);
    }
    return newlineTokenIds;
}

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

    // Sum log-probs for each forced token.
    // Note: in autoregressive transformers, logits at index (pos) predict token at (pos + 1).
    // The first token of tokenIds is at index startPos, so its logits are at startPos - 1.
    let logProb = 0;
    const startPos = seqLen - tokenIds.length;
    for (let i = 0; i < tokenIds.length; i++) {
        const pos = Math.max(0, startPos - 1 + i);
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
 * Checks:
 * 1. Must not be in the middle of an elision or hyphen.
 * 2. partialVerse must have at least 7 syllables (or >= 24 chars) so the verse
 *    reaches 10-12 syllables when the rhyme word is appended.
 * 3. Exact candidates are filtered and sorted primarily by meter adherence (target 11),
 *    and scored with the model's teacher-forced log probability.
 */
function isReadyForWordAppend(text) {
    if (!text) return true;
    const trimmed = text.trimEnd();
    if (!trimmed) return true;
    if (trimmed.endsWith("'") || trimmed.endsWith("-")) return false;
    // Lone single consonant at end of line (e.g. " faceva s") means an incomplete subword
    if (/\s+[b-df-hj-np-tv-z]$/i.test(trimmed)) return false;
    return true;
}

async function findForcedRhymeCompletion(context, partialVerse, targetSuffix,
    isSearchCurrent = () => true) {
    if (!rimario || !targetSuffix) return null;

    // Must not be in the middle of an incomplete word, apostrophe, or hyphen
    if (!isReadyForWordAppend(partialVerse)) return null;

    const currentSyllables = countItalianSyllables(partialVerse);
    if (currentSyllables < 7 && partialVerse.length < 24) return null;

    // Get candidate rhyme words from rimario
    const exactWords = getRhymeFamilies(targetSuffix);
    if (!exactWords.length) return null;

    // Filter out already-used rhyme words in this canto if possible
    const availableExact = exactWords.filter(w => !usedRhymeWords.has(w));
    const pool = availableExact.length > 0 ? availableExact : exactWords;

    const prefixSpace = partialVerse.endsWith(' ') ? '' : ' ';

    // Pre-calculate meter deviation for all candidates
    const candidates = [];
    for (const word of pool) {
        const fullLine = partialVerse + prefixSpace + word;
        const syllables = countItalianSyllables(fullLine);
        const meterDev = Math.abs(syllables - 11);
        candidates.push({
            word,
            fullLine,
            syllables,
            meterDev
        });
    }

    // Filter by meter: prefer meterDev === 0 (exactly 11), then meterDev <= 1 (10 or 12)
    let viable = candidates.filter(c => c.meterDev === 0);
    if (!viable.length) {
        viable = candidates.filter(c => c.meterDev <= 1);
    }
    if (!viable.length) {
        viable = candidates.slice(0, 10);
    }

    // Score top candidates with the model (cap at 6 for rapid responsiveness)
    const toScore = viable.slice(0, 6);
    const scored = [];

    for (let i = 0; i < toScore.length; i++) {
        if (!isSearchCurrent()) return { canceled: true };
        const cand = toScore[i];
        const tokenIds = encode(prefixSpace + cand.word + '\n');
        const logProb = await scoreTokenSequence(context, tokenIds);
        scored.push({
            ...cand,
            tokenIds,
            logProb
        });
    }

    if (!scored.length) return null;

    // Rank candidates: model log-probability + bonus for exact 11 syllables
    scored.sort((a, b) => {
        const aMeterBonus = a.meterDev === 0 ? 3.0 : (a.meterDev === 1 ? 1.0 : -5.0);
        const bMeterBonus = b.meterDev === 0 ? 3.0 : (b.meterDev === 1 ? 1.0 : -5.0);
        return (b.logProb + bMeterBonus) - (a.logProb + aMeterBonus);
    });

    const best = scored[0];
    const level = best.meterDev === 0 ? 'exact+11' : (best.meterDev === 1 ? 'exact+10-12' : 'forced-insert');
    return buildForcedResult(best, level, prefixSpace);
}

function buildForcedResult(candidate, level, prefixSpace = ' ') {
    const ending = getEndingSound(candidate.fullLine);
    return {
        tokenIds: candidate.tokenIds,
        text: prefixSpace + candidate.word + '\n',
        logProbability: candidate.logProb,
        rank: candidate.logProb,
        validation: {
            accepted: true,
            line: candidate.fullLine,
            ending,
            meterScore: getVerseMeterScore(candidate.fullLine),
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

    // Safety cutoff: never allow any verse to exceed 48 characters (Dante average is ~36, max 53)
    if (partialVerse.length >= 45) {
        if (targetEnding && rimario) {
            const urgent = await findForcedRhymeCompletion(context, partialVerse, targetEnding, () => true);
            if (urgent && !urgent.canceled && urgent.tokenIds) {
                pendingRhymeTokenQueue = urgent.tokenIds.slice();
                usedRhymeWords.add(urgent.validation.word);
                console.log(`[RHYME] URGENT FORCED selected "${urgent.validation.word}" for target "${targetEnding}"`);
                return pendingRhymeTokenQueue.shift();
            }
        }
        // Force newline to end verse and preserve meter
        const nlTokens = encode('\n');
        return nlTokens[0];
    }

    // Suppress premature newline token if verse is in rhyming mode and hasn't reached minimum length
    if (danteRhymeMode !== RHYME_MODE_OFF && targetEnding) {
        const sylCount = countItalianSyllables(partialVerse);
        if (sylCount < 7 && partialVerse.length < 24) {
            const nlSet = getNewlineTokenIds();
            let suppressed = false;
            for (const nlId of nlSet) {
                if (probs[nlId] > 0) {
                    probs[nlId] = 0;
                    suppressed = true;
                }
            }
            if (suppressed) {
                let sum = 0;
                for (let i = 0; i < probs.length; i++) sum += probs[i];
                if (sum > 0) {
                    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
                }
            }
        }
    }

    // FORCED mode: use rimario-based completion
    if (danteRhymeMode === RHYME_MODE_FORCED && targetEnding) {
        const sylCount = countItalianSyllables(partialVerse);
        const atWordBoundary = isReadyForWordAppend(partialVerse);
        const canTrigger = atWordBoundary && (sylCount >= 7 || partialVerse.length >= 24);

        if (canTrigger) {
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
        }
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
