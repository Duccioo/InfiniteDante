const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {
    countItalianSyllables,
    decodeTokenByteSequences,
    doTheyRhyme,
    evaluateCompletedRhymeContinuation,
    fallbackToNormalSampling,
    getAssonanceScore,
    getCompletedLineEndings,
    getExactEndingProgressScore,
    getEndingSound,
    getRhymeTarget,
    isRhymeSearchSnapshotCurrent
} = require('./rhyme.js');

const { isWordBoundary } = require('./rhyme.js');

assert.strictEqual(getEndingSound('La nostra vita!'), 'ita');
assert.strictEqual(getEndingSound('L\'amor è'), 'e');
assert.strictEqual(getEndingSound('perché'), 'e');
assert.strictEqual(getEndingSound('occhio'), 'occhio');

// Phonetic & Dante Rhyme tests
assert.strictEqual(doTheyRhyme(getEndingSound('canto'), getEndingSound('vento')), false,
    'matching only the final two letters is not a strict rhyme');
assert.strictEqual(doTheyRhyme(getEndingSound('vita'), getEndingSound('smarrita')), true);
assert.strictEqual(doTheyRhyme(getEndingSound('oscura'), getEndingSound('paura')), true,
    'oscura and paura must rhyme in Dante (hiatus pa-u-ra)');
assert.strictEqual(doTheyRhyme(getEndingSound('dura'), getEndingSound('paura')), true);
assert.strictEqual(doTheyRhyme(getEndingSound('trovai'), getEndingSound('intrai')), true,
    'trovai and intrai must rhyme on oxytone -ai');
assert.strictEqual(doTheyRhyme(getEndingSound('desio'), getEndingSound('oblio')), true,
    'desio and oblio must rhyme on tonic -io');
assert.strictEqual(doTheyRhyme(getEndingSound('punto'), getEndingSound('giunto')), true,
    'punto and giunto must rhyme on -unto (diacritic i)');

assert.ok(getExactEndingProgressScore('Nel mezzo del cammin di nostra vita', 'ita') >
    getExactEndingProgressScore('Nel mezzo del cammin di nostra vento', 'ita'),
    'an exact assembled ending must outrank non-rhyme progress');
assert.ok(getAssonanceScore('asa', 'ana') > 0, 'assonance remains available as a weak hint');
assert.strictEqual(doTheyRhyme('asa', 'ana'), false, 'assonance must not satisfy strict rhyme');

assert.deepStrictEqual([0, 1, 2, 3, 4, 5, 6, 7].map(getRhymeTarget),
    [-1, -1, 0, 1, -1, 3, 4, -1]);

// Meter tests with sinalefe
assert.strictEqual(countItalianSyllables('Nel mezzo del cammin di nostra vita'), 11);
assert.strictEqual(countItalianSyllables('mi ritrovai per una selva oscura'), 11,
    'sinalefe between selva and oscura must produce 11 syllables');
assert.strictEqual(countItalianSyllables('ché la diritta via era smarrita'), 11,
    'sinalefe between via and era must produce 11 syllables');
assert.strictEqual(countItalianSyllables('Ahi quanto a dir qual era è cosa dura'), 11);

// Word boundary tests
assert.strictEqual(isWordBoundary('Nel mezzo '), true);
assert.strictEqual(isWordBoundary('Nel mezzo'), false);
assert.strictEqual(isWordBoundary('selva,'), true);


const continuation = decodeTokenByteSequences([
    new Uint8Array([0x76, 0x69]), // vi
    new Uint8Array([0x74, 0x61, 0x0a]) // ta\n
]);
assert.strictEqual(continuation, 'vita\n');
assert.strictEqual(decodeTokenByteSequences([
    new Uint8Array([0xc3]), new Uint8Array([0xa8])
]), 'è', 'split UTF-8 bytes must be assembled before decoding');
const pendingUtf8Snapshot = new Uint8Array([0xc3]);
assert.strictEqual(decodeTokenByteSequences([
    new Uint8Array([0xa8, 0x0a])
], pendingUtf8Snapshot), 'è\n',
'a pending UTF-8 prefix must decode with the first candidate token exactly once');
assert.deepStrictEqual(Array.from(pendingUtf8Snapshot), [0xc3], 'candidate decoding must not mutate its snapshot');

const completed = evaluateCompletedRhymeContinuation(
    'Nel mezzo del cammin di nostra ', continuation, getEndingSound('vita')
);
assert.strictEqual(completed.accepted, true);
assert.strictEqual(completed.ending, 'ita');
assert.strictEqual(evaluateCompletedRhymeContinuation(
    'Nel mezzo del cammin di nostra ', 'vita', 'ita'
).reason, 'no-newline');
assert.strictEqual(evaluateCompletedRhymeContinuation(
    'Nel mezzo del cammin di nostra ', 'vento\n', 'ita'
).reason, 'not-a-rhyme');
assert.strictEqual(evaluateCompletedRhymeContinuation(
    'Nel mezzo del cammin di nostra ', 'rana\n', 'asa'
).reason, 'not-a-rhyme', 'assonance cannot accept a completed STRICT line');
assert.strictEqual(evaluateCompletedRhymeContinuation(
    'Nel mezzo del cammin di nostra ', 'vita\nuna seconda riga\n', 'ita'
).reason, 'multiple-newlines');

assert.deepStrictEqual(getCompletedLineEndings('old\n', '\nprefix\n\ntrailing'),
    ['', 'prefix', ''], 'every newline completes exactly its own line, including empty lines');
assert.deepStrictEqual(getCompletedLineEndings('old\npref', 'ix\nnext'), ['prefix'],
    'a post-newline prefix remains part of the next generated line');

const snapshot = { epoch: 4, mode: 'STRICT', verseNumber: 3, contextKey: '1,2', generatedText: 'line' };
assert.strictEqual(isRhymeSearchSnapshotCurrent(snapshot, { ...snapshot }), true);
assert.strictEqual(isRhymeSearchSnapshotCurrent(snapshot, { ...snapshot, epoch: 5 }), false,
    'an in-flight result cannot survive cancellation or a mode change');
assert.strictEqual(isRhymeSearchSnapshotCurrent(snapshot, { ...snapshot, mode: 'OFF' }), false);
assert.strictEqual(fallbackToNormalSampling([0.1, 0.9], () => 0), 0,
    'STRICT failure must delegate to normal sampling rather than choose argmax');

const stateSource = fs.readFileSync(path.join(__dirname, 'state.js'), 'utf8');
const stateContext = {};
vm.createContext(stateContext);
vm.runInContext(`${stateSource}
globalThis.__stateTestApi = {
    beginGenerationRun,
    cancelPendingRhymeCompletion,
    invalidateGenerationRun,
    isGenerationRunCurrent,
    resetRhymeSearchAttempts,
    shouldHandleGenerationRunError,
    get: () => ({ attempts: rhymeSearchAttemptCount, lastLength: rhymeSearchLastAttemptLength,
        queue: pendingRhymeTokenQueue.slice(), epoch: rhymeGenerationEpoch }),
    set: (attempts, lastLength, queue) => {
        rhymeSearchAttemptCount = attempts;
        rhymeSearchLastAttemptLength = lastLength;
        pendingRhymeTokenQueue = queue;
    },
    setGenerating: value => { isGenerating = value; }
};`, stateContext);
const stateApi = stateContext.__stateTestApi;
stateApi.set(2, 34, [9, 10]);
const beforeCancel = stateApi.get();
stateApi.cancelPendingRhymeCompletion(); // mode change
const afterModeChange = stateApi.get();
assert.strictEqual(afterModeChange.queue.length, 0);
assert.strictEqual(afterModeChange.attempts, 2);
assert.strictEqual(afterModeChange.lastLength, 34);
assert.strictEqual(afterModeChange.epoch, beforeCancel.epoch + 1);
stateApi.cancelPendingRhymeCompletion(); // pause/resume cancellation
const afterPause = stateApi.get();
assert.strictEqual(afterPause.attempts, 2);
assert.strictEqual(afterPause.lastLength, 34);
stateApi.resetRhymeSearchAttempts(); // completed newline or new verse
const afterVerseReset = stateApi.get();
assert.strictEqual(afterVerseReset.attempts, 0);
assert.strictEqual(afterVerseReset.lastLength, -Infinity);
assert.strictEqual(afterVerseReset.queue.length, 0);
assert.strictEqual(afterVerseReset.epoch, afterPause.epoch);
stateApi.setGenerating(true);
const oldRun = stateApi.beginGenerationRun();
assert.strictEqual(stateApi.isGenerationRunCurrent(oldRun), true);
stateApi.setGenerating(false); // stop
stateApi.invalidateGenerationRun();
stateApi.setGenerating(true); // immediate new start
const newRun = stateApi.beginGenerationRun();
assert.strictEqual(stateApi.isGenerationRunCurrent(oldRun), false,
    'the old loop fails the pre-call generateNext guard after a new run starts');
assert.strictEqual(stateApi.isGenerationRunCurrent(newRun), true);
assert.strictEqual(stateApi.shouldHandleGenerationRunError(oldRun), false,
    'an obsolete run rejection must not stop the newly started run');
assert.strictEqual(stateApi.shouldHandleGenerationRunError(newRun), true);

stateApi.set(2, 34, []);
const staleInvocation = {
    epoch: stateApi.get().epoch,
    mode: 'STRICT',
    verseNumber: 3,
    contextKey: '1,2',
    generatedText: 'line'
};
stateApi.cancelPendingRhymeCompletion(); // stop/restart while initial inference awaits
const currentInvocation = { ...staleInvocation, epoch: stateApi.get().epoch };
if (isRhymeSearchSnapshotCurrent(staleInvocation, currentInvocation)) {
    stateApi.set(3, 40, []); // This models generateNext's attempt mutation.
}
assert.strictEqual(stateApi.get().attempts, 2,
    'a stale invocation is rejected before it can mutate rhyme attempts after initial inference');

const tokenizerSource = fs.readFileSync(path.join(__dirname, 'tokenizer.js'), 'utf8');
const tokenizerContext = { TextEncoder, TextDecoder, Uint8Array, Object, Array };
vm.createContext(tokenizerContext);
vm.runInContext(`${tokenizerSource}
initBPE({});
globalThis.__tokenizerTestApi = { decode, resetDecoder };`, tokenizerContext);
const tokenizerApi = tokenizerContext.__tokenizerTestApi;
tokenizerApi.decode([0xc3]); // pending lead byte from an abandoned run
tokenizerApi.resetDecoder(); // start/resume reconstructs visible text before encoding
assert.strictEqual(tokenizerApi.decode([0x41]), 'A',
    'resume reset prevents an abandoned UTF-8 lead byte from corrupting ASCII output');

(async () => {
    const inferenceSource = fs.readFileSync(path.join(__dirname, 'inference.js'), 'utf8');
    const inferenceContext = {
        Array,
        BigInt,
        BigInt64Array,
        Float32Array,
        Map,
        Math,
        Number,
        Object,
        Uint8Array,
        bpe_vocab: [new Uint8Array([97]), new Uint8Array([98]), new Uint8Array([99])],
        danteRhymeMode: 'STRICT',
        effectiveBlockSize: 16,
        RHYME_MODE_SOFT: 'SOFT',
        decodeTokenByteSequences: chunks => String.fromCharCode(...chunks.map(chunk => chunk[0])),
        evaluateCompletedRhymeContinuation: () => ({ accepted: false, reason: 'no-newline' }),
        getAssonanceScore: () => 0,
        getExactEndingProgressScore: () => 0
    };
    vm.createContext(inferenceContext);
    vm.runInContext(`${inferenceSource}
globalThis.__inferenceTestApi = { findRhymeCompletion };`, inferenceContext);

    let branchCalls = 0;
    let searchCurrent = true;
    const canceled = await inferenceContext.__inferenceTestApi.findRhymeCompletion(
        [0], [0.5, 0.3, 0.2], 'partial ', 'ita', new Uint8Array(0),
        () => searchCurrent,
        async () => {
            branchCalls++;
            searchCurrent = false;
            return [0.5, 0.3, 0.2];
        }
    );
    assert.strictEqual(canceled.canceled, true);
    assert.strictEqual(branchCalls, 1,
        'cancellation after the first branch inference prevents remaining ONNX calls');

    console.log('rhyme helper tests passed');
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
