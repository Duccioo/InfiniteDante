/**
 * rhyme.js
 * =========
 * Dante terza-rima helpers with rimario-based rhyme validation.
 *
 * Provides:
 * - Accurate Italian poetic phonetics & rhyme suffix extraction
 * - Metric syllable scansion with sinalefe
 * - Rimario loading and index management
 * - Rhyme candidate evaluation and meter scoring
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

// Explicit known hiatus stems where stress is on the hiatus vowel
const KNOWN_HIATUS_STEMS = {
    'paura': 'ura',
    'paure': 'ure',
};

// Words ending in tonic -io (de-sì-o, o-blì-o, r-ì-o, na-tì-o, add-ì-o)
const TONIC_IO_WORDS = new Set([
    'desio', 'oblio', 'rio', 'natio', 'mormorio', 'gentilio', 'pendio', 'addio', 
    'fio', 'disio', 'campio', 'brischio', 'restio', 'brusio', 'ronzio', 'calpestio'
]);

// Words ending in tonic -ia (v-ì-a, pr-ì-a, m-ì-a, s-ì-a, com-pa-gn-ì-a, fol-l-ì-a, ecc.)
const TONIC_IA_WORDS = new Set([
    'via', 'pria', 'mia', 'sia', 'follia', 'armonia', 'poesia', 'cortesia',
    'allegria', 'compagnia', 'balia', 'ria', 'magia', 'fantasia', 'ironia',
    'bugia', 'corsia', 'fabbria', 'malia', 'villania', 'gelosia', 'fiancheria',
    'fantesia', 'baratteria', 'idropesia', 'epia', 'abbadia', 'dia', 'disvia',
    'invia', 'devia', 'godia', 'udia', 'sentia', 'partia', 'uscia', 'salia',
    'smarria', 'maria', 'girolomia', 'tenia', 'vedia', 'faria', 'daria', 'saria',
    'avria', 'poria', 'credea', 'solia', 'arderia', 'valia', 'pazia'
]);

const NON_TONIC_IA_WORDS = new Set([
    'grazia', 'audacia', 'minaccia', 'angoscia', 'striscia', 'lascia', 'faccia',
    'provincia', 'materia', 'memoria', 'gloria', 'storia', 'vittoria', 'notizia',
    'sentenzia', 'ingiuria', 'ignominia', 'calunnia', 'curia', 'furia'
]);

function normalizeItalian(text) {
    return String(text || '')
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '');
}

/**
 * Split a word into syllable nuclei (vowels/diphthongs).
 * Each nucleus is [startPos, endPos]. Diphthongs merge into one nucleus;
 * accented vowels force a hiatus.
 * Diacritic 'i' after c/g before a/o/u (giunto, gioco) is skipped.
 */
function splitSyllableNuclei(word) {
    let plain = Array.from(word).map(c => ACCENTED_MAP[c] || c).join('').toLowerCase();
    // Silent 'h' in Italian is never pronounced (ahi -> ai, ho -> o, etc.)
    plain = plain.replace(/h/g, '');
    const nuclei = [];
    let i = 0;
    while (i < plain.length) {

        if (ALL_VOWELS.includes(plain[i])) {
            // Check diacritic 'i' after c/g before a/o/u
            if (plain[i] === 'i' && i > 0 && (plain[i-1] === 'c' || plain[i-1] === 'g') &&
                i + 1 < plain.length && (plain[i+1] === 'a' || plain[i+1] === 'o' || plain[i+1] === 'u')) {
                i++;
                continue;
            }

            const start = i;
            while (i + 1 < plain.length &&
                ALL_VOWELS.includes(plain[i + 1]) &&
                DIPHTHONGS_SET.has(plain[i] + plain[i + 1]) &&
                !ACCENTED_VOWELS_SET.has(word[i]) &&
                !ACCENTED_VOWELS_SET.has(word[i + 1])) {
                const pair = plain[i] + plain[i + 1];
                if (pair === 'ia' || pair === 'io' || pair === 'ie' || pair === 'ea' || pair === 'eo' || pair === 'oa') {
                    const cleanW = word.toLowerCase();
                    if (TONIC_IO_WORDS.has(cleanW) || TONIC_IA_WORDS.has(cleanW)) {
                        break;
                    }
                }
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
 * 1. Check known explicit poetic hiatus words (paura -> ura, desio -> io, via -> ia).
 * 2. If the word contains an accented vowel, stress is there.
 * 3. Check oxytone diphthong endings (-ai, -ei, -ui, -oi: trovai -> ai).
 * 4. Check hiatus pattern: -aura -> -ura (paura).
 * 5. Otherwise assume paroxytone (stress on penultimate syllable nucleus).
 * 6. Accents are normalized to plain vowels in the returned suffix.
 */
function getStressedSuffix(word) {
    const w = String(word || '').toLowerCase().replace(/[^a-zàèéìíòóùú]/g, '');
    if (w.length < 2) return ACCENTED_MAP[w] || w;

    // 1. Explicit stems and whole words
    if (KNOWN_HIATUS_STEMS[w]) return KNOWN_HIATUS_STEMS[w];
    if (TONIC_IO_WORDS.has(w)) return 'io';
    if (TONIC_IA_WORDS.has(w) || (w.endsWith('ia') && w.length <= 4)) return 'ia';
    if (w.endsWith('ia') && w.length >= 4) {
        if (!NON_TONIC_IA_WORDS.has(w) && (w.endsWith('ria') || w.endsWith('dia') || w.endsWith('tia') || w.endsWith('via') || w.endsWith('nia') || w.endsWith('lia'))) {
            return 'ia';
        }
    }

    // 2. Explicit accent
    for (let i = 0; i < w.length; i++) {
        if (ACCENTED_VOWELS_SET.has(w[i])) {
            let suffix = '';
            for (let j = i; j < w.length; j++) {
                suffix += ACCENTED_MAP[w[j]] || w[j];
            }
            return suffix;
        }
    }

    // 3. Oxytone (tronca) verbal endings: -ai, -ei, -ui, -oi
    if (w.length >= 3 && (w.endsWith('ai') || w.endsWith('ei') || w.endsWith('ui') || w.endsWith('oi'))) {
        return w.slice(-2);
    }

    // 4. Hiatus pattern: -aura (paura -> ura)
    if (w.endsWith('aura') && w !== 'aura' && !w.endsWith('laura')) {
        return 'ura';
    }

    // 5. Penultimate syllable nucleus (diphthong-aware)
    const nuclei = splitSyllableNuclei(w);
    if (!nuclei.length) return w.slice(-3);

    const stressStart = nuclei.length >= 2 ? nuclei[nuclei.length - 2][0] : nuclei[0][0];
    return w.slice(stressStart);
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
 * Count syllables in a single word.
 */
function countWordSyllables(word) {
    word = String(word || '').toLowerCase().replace(/[^a-zàèéìíòóùú']/g, '');
    if (!word) return 0;
    const nuclei = splitSyllableNuclei(word);
    return Math.max(1, nuclei.length);
}

/**
 * Count Italian poetic syllables in a line of poetry, applying SINALEFE.
 * Sinalefe: elision between a word ending in a vowel (or apostrophe)
 * and the next word beginning with a vowel.
 */
function countItalianSyllables(text) {
    text = String(text || '').trim();
    if (!text) return 0;
    const words = text.match(/[a-zàèéìíòóùú']+/gi) || [];
    if (!words.length) return 0;

    const VOWEL_RE = /[aeiouàèéìíòóùú]/i;
    let total = 0;

    for (let i = 0; i < words.length; i++) {
        const w = words[i].toLowerCase();
        total += countWordSyllables(w);

        // Sinalefe between word i and word i + 1
        if (i < words.length - 1) {
            const nextW = words[i + 1].toLowerCase();
            const currEndsVowel = w.endsWith("'") || VOWEL_RE.test(w.slice(-1));
            const nextStartsVowel = VOWEL_RE.test(nextW[0]);
            if (currEndsVowel && nextStartsVowel) {
                total -= 1; // Sinalefe: vowels merge into one metric syllable
            }
        }
    }
    return Math.max(1, total);
}

/**
 * Score line meter adherence to Italian endecasillabo (11 syllables).
 */
function getVerseMeterScore(line) {
    const syllables = countItalianSyllables(line);
    if (syllables < 8 || syllables > 15) return 0;
    const deviation = Math.abs(syllables - 11);
    if (deviation === 0) return 1.0;
    if (deviation === 1) return 0.8;
    if (deviation === 2) return 0.5;
    return 0.2;
}

/**
 * Check if the given text currently ends on a complete word boundary
 * (i.e. whitespace, punctuation, or newline — not mid-word or mid-BPE-token).
 */
function isWordBoundary(text) {
    if (!text) return true;
    return /[\s.,;:!?…—–]$/.test(text) || text.endsWith('\n');
}

/**
 * Load rimario.json and build the inverse index (word -> suffix).
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
 */
function evaluateCompletedRhymeContinuation(partialVerse, continuation, targetEnding) {
    const firstNewline = continuation.indexOf('\n');
    if (firstNewline < 0) return { accepted: false, reason: 'no-newline' };
    if (continuation.indexOf('\n', firstNewline + 1) >= 0) {
        return { accepted: false, reason: 'multiple-newlines' };
    }

    const lineEnd = continuation.slice(0, firstNewline);
    const line = `${partialVerse}${lineEnd}`;

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
        countWordSyllables,
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
