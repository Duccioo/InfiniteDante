/**
 * rhyme.js
 * =========
 * Dante terza-rima helpers with rimario-based rhyme validation.
 *
 * Key improvement over the previous version: rhyme suffixes are extracted
 * from the stressed (tonic) vowel — not the penultimate vowel character —
 * using syllable-aware nucleus splitting.  When rimario.json is loaded,
 * known words get their exact suffix from the dictionary.
 */

// Italian diphthongs (count as one syllable nucleus)
const DIPHTHONGS_SET = new Set([
    'ia', 'ie', 'io', 'iu',
    'ua', 'ue', 'ui', 'uo',
    'ai', 'ei', 'oi', 'au', 'eu',
]);

const ACCENTED_MAP = {
    'à': 'a', 'è': 'e', 'é': 'e', 'ì': 'i', 'í': 'i',
    'ò': 'o', 'ó': 'o', 'ù': 'u', 'ú': 'u',
};
const ACCENTED_VOWELS_SET = new Set(Object.keys(ACCENTED_MAP));
const ALL_VOWELS = 'aeiouàèéìíòóùú';

function normalizeItalian(text) {
    return String(text || '')
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '');
}

/**
 * Split a word into syllable nuclei (vowels/diphthongs).
 * Each nucleus is [startPos, endPos].  Diphthongs merge into one nucleus;
 * accented vowels force a hiatus.
 */
function splitSyllableNuclei(word) {
    const plain = Array.from(word).map(c => ACCENTED_MAP[c] || c).join('');
    const nuclei = [];
    let i = 0;
    while (i < plain.length) {
        if (ALL_VOWELS.includes(plain[i])) {
            const start = i;
            while (i + 1 < plain.length &&
                ALL_VOWELS.includes(plain[i + 1]) &&
                DIPHTHONGS_SET.has(plain[i] + plain[i + 1]) &&
                !ACCENTED_VOWELS_SET.has(word[i]) &&
                !ACCENTED_VOWELS_SET.has(word[i + 1])) {
                i++;
            }
            nuclei.push([start, i]);
            i++;
        } else {
            i++;
        }
    }
    return nuclei;
}

/**
 * Extract the rhyme suffix from the stressed vowel to end of word.
 *
 * 1. If the word contains an accented vowel, stress is there.
 * 2. Otherwise assume paroxytone (stress on penultimate syllable nucleus).
 * 3. Accents are normalized to plain vowels in the returned suffix.
 */
function getStressedSuffix(word) {
    word = String(word || '').toLowerCase().replace(/[^a-zàèéìíòóùú]/g, '');
    if (word.length < 2) return ACCENTED_MAP[word] || word;

    // 1. Explicit accent
    for (let i = 0; i < word.length; i++) {
        if (ACCENTED_VOWELS_SET.has(word[i])) {
            let suffix = '';
            for (let j = i; j < word.length; j++) {
                suffix += ACCENTED_MAP[word[j]] || word[j];
            }
            return suffix;
        }
    }

    // 2. Penultimate syllable nucleus (diphthong-aware)
    const nuclei = splitSyllableNuclei(word);
    if (!nuclei.length) return word.slice(-3);

    const stressStart = nuclei.length >= 2 ? nuclei[nuclei.length - 2][0] : nuclei[0][0];
    return word.slice(stressStart);
}

/**
 * Return the rhyme suffix of the final word in a verse.
 * If the rimario is loaded (rimarioIndex), uses the known suffix;
 * otherwise falls back to the heuristic.
 */
function getEndingSound(text) {
    const cleaned = String(text || '').toLowerCase().trim().replace(/[^a-zàèéìíòóùú ]+$/g, '');
    const match = cleaned.match(/[a-zàèéìíòóùú]+$/);
    const lastWord = match ? match[0] : '';
    if (!lastWord) return '';

    // Check rimario index first
    if (typeof rimarioIndex !== 'undefined' && rimarioIndex instanceof Map && rimarioIndex.size > 0) {
        const known = rimarioIndex.get(lastWord);
        if (known) return known;
    }

    return getStressedSuffix(lastWord);
}

/**
 * Strict rhyme predicate: complete stressed suffixes must match.
 */
function doTheyRhyme(ending1, ending2) {
    const first = normalizeItalian(ending1).replace(/[^a-z]+/g, '');
    const second = normalizeItalian(ending2).replace(/[^a-z]+/g, '');
    return first.length >= 2 && first === second;
}

/** A strong search-progress signal; it is not a completed-line acceptance. */
function getExactEndingProgressScore(line, targetEnding) {
    return doTheyRhyme(getEndingSound(line), targetEnding) ? 1 : 0;
}

/**
 * A weaker signal for SOFT ranking only; never use this to assert a rhyme.
 */
function getAssonanceScore(ending1, ending2) {
    const first = normalizeItalian(ending1).replace(/[^a-z]/g, '');
    const second = normalizeItalian(ending2).replace(/[^a-z]/g, '');
    if (!first || !second) return 0;
    if (first === second) return 1;

    const vowels = value => (value.match(/[aeiou]/g) || []).join('');
    const firstVowels = vowels(first);
    const secondVowels = vowels(second);
    if (firstVowels.length >= 2 && firstVowels === secondVowels) return 0.45;

    let suffix = 0;
    while (suffix < Math.min(first.length, second.length) &&
        first[first.length - suffix - 1] === second[second.length - suffix - 1]) {
        suffix++;
    }
    return suffix >= 3 ? Math.min(0.4, suffix / Math.max(first.length, second.length)) : 0;
}

/**
 * Approximate vowel-group syllables for ranking and broad plausibility guards.
 */
function countItalianSyllables(text) {
    const words = normalizeItalian(text).match(/[a-z]+/g) || [];
    let count = 0;
    for (const word of words) {
        const groups = word.match(/[aeiou]+/g);
        count += groups ? groups.length : 0;
    }
    return count;
}

function getVerseMeterScore(line) {
    const syllables = countItalianSyllables(line);
    if (syllables < 5 || syllables > 18) return 0;
    return Math.max(0, 1 - Math.abs(syllables - 11) / 10);
}

/**
 * Check that text ends on a word boundary (the last word is complete,
 * not a BPE fragment).  Accepts: word + optional punctuation + newline.
 */
function isWordBoundary(text) {
    return /[a-zàèéìíòóùú][.,;:!?…]*\n?$/.test(text.trimEnd());
}

/**
 * Load rimario.json and build the inverse index (word → suffix).
 * Returns a promise that resolves to the rimario object.
 */
async function loadRimario(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        rimario = data;
        rimarioIndex = new Map();
        for (const [suffix, words] of Object.entries(data)) {
            for (const word of words) {
                rimarioIndex.set(word.toLowerCase(), suffix);
            }
        }
        console.log(`[RIMARIO] Loaded ${rimarioIndex.size} words in ${Object.keys(data).length} rhyme families`);
        return data;
    } catch (error) {
        console.warn(`[RIMARIO] Failed to load rimario: ${error.message}`);
        rimario = null;
        rimarioIndex = new Map();
        return null;
    }
}

/**
 * Get all words that rhyme with the given suffix from the rimario.
 * Returns an empty array if rimario is not loaded or suffix not found.
 */
function getRhymeFamilies(suffix) {
    if (!rimario || !suffix) return [];
    return rimario[suffix] || [];
}

/** Decode BPE byte chunks together, preserving multi-token UTF-8 characters. */
function decodeTokenByteSequences(byteSequences, pendingPrefix = new Uint8Array(0)) {
    const chunks = byteSequences.filter(Boolean);
    const prefix = pendingPrefix || new Uint8Array(0);
    const length = prefix.length + chunks.reduce((total, chunk) => total + chunk.length, 0);
    const bytes = new Uint8Array(length);
    bytes.set(prefix, 0);
    let offset = prefix.length;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    // Mirror tokenizer.js: trailing incomplete UTF-8 belongs to the next token.
    let validEnd = bytes.length;
    for (let index = Math.max(0, bytes.length - 4); index < bytes.length; index++) {
        const byte = bytes[index];
        if ((byte & 0x80) === 0) {
            validEnd = index + 1;
        } else if ((byte & 0xE0) === 0xC0) {
            if (index + 2 <= bytes.length) validEnd = index + 2;
            else { validEnd = index; break; }
        } else if ((byte & 0xF0) === 0xE0) {
            if (index + 3 <= bytes.length) validEnd = index + 3;
            else { validEnd = index; break; }
        } else if ((byte & 0xF8) === 0xF0) {
            if (index + 4 <= bytes.length) validEnd = index + 4;
            else { validEnd = index; break; }
        }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes.slice(0, validEnd));
}

/** Return every line completed by emitted text, including deliberately empty lines. */
function getCompletedLineEndings(previousText, emittedText) {
    let line = String(previousText || '').split('\n').pop();
    const completed = [];
    for (const character of String(emittedText || '')) {
        if (character === '\n') {
            completed.push(line);
            line = '';
        } else {
            line += character;
        }
    }
    return completed;
}

function isRhymeSearchSnapshotCurrent(snapshot, current) {
    return snapshot.epoch === current.epoch && snapshot.mode === current.mode &&
        snapshot.verseNumber === current.verseNumber && snapshot.contextKey === current.contextKey &&
        snapshot.generatedText === current.generatedText;
}

function fallbackToNormalSampling(probs, sampler) {
    return sampler(probs);
}

/**
 * Validate a decoded continuation only once it has ended the current line.
 * A candidate may contain text after its one newline, but never a second line.
 *
 * KEY IMPROVEMENT: the last word before the newline must be a complete word
 * (checked via isWordBoundary), not a BPE fragment.
 */
function evaluateCompletedRhymeContinuation(partialVerse, continuation, targetEnding) {
    const firstNewline = continuation.indexOf('\n');
    if (firstNewline < 0) return { accepted: false, reason: 'no-newline' };
    if (continuation.indexOf('\n', firstNewline + 1) >= 0) {
        return { accepted: false, reason: 'multiple-newlines' };
    }

    const lineEnd = continuation.slice(0, firstNewline);
    const line = `${partialVerse}${lineEnd}`;

    // Verify the line ends on a word boundary, not mid-BPE-token
    if (lineEnd.length > 0 && !/[a-zàèéìíòóùú]/.test(lineEnd.slice(-1).replace(/[.,;:!?…]/g, ''))) {
        return { accepted: false, reason: 'no-word-ending', line };
    }

    const ending = getEndingSound(line);
    const meterScore = getVerseMeterScore(line);
    if (meterScore === 0) return { accepted: false, reason: 'implausible-meter', line, ending };
    if (!doTheyRhyme(ending, targetEnding)) {
        return { accepted: false, reason: 'not-a-rhyme', line, ending, meterScore };
    }
    return { accepted: true, line, ending, meterScore };
}

/** Get the terza-rima target verse (ABA BCB CDC...). */
function getRhymeTarget(verseIndex) {
    if (verseIndex < 2 || verseIndex % 3 === 1) return -1;
    return verseIndex - 2;
}

function getCurrentPartialVerse() {
    const lines = generatedText.split('\n');
    return lines[lines.length - 1] || '';
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        countItalianSyllables,
        decodeTokenByteSequences,
        doTheyRhyme,
        evaluateCompletedRhymeContinuation,
        fallbackToNormalSampling,
        getAssonanceScore,
        getCompletedLineEndings,
        getEndingSound,
        getExactEndingProgressScore,
        getRhymeFamilies,
        getRhymeTarget,
        getStressedSuffix,
        getVerseMeterScore,
        isRhymeSearchSnapshotCurrent,
        isWordBoundary,
        normalizeItalian,
        splitSyllableNuclei,
    };
}
