/**
 * rhyme.js
 * ========
 * Dante terza rima (ABA BCB CDC...) rhyme logic.
 */

// ============================================================================
// Rhyme Functions
// ============================================================================

/**
 * Get the ending sound pattern of a word/verse for rhyme matching.
 * Extracts the last vowel cluster and following consonants.
 */
function getEndingSound(text) {
    // Clean and normalize the text
    const cleaned = text.toLowerCase().trim();
    if (cleaned.length === 0) return '';
    
    // Italian vowels
    const vowels = 'aeiouàèéìòóù';
    
    // Find the last accented/stressed syllable pattern
    // Look for the last 2-4 characters that form a rhyme pattern
    let ending = '';
    let foundVowel = false;
    
    for (let i = cleaned.length - 1; i >= 0 && ending.length < 5; i--) {
        const char = cleaned[i];
        if (char === ' ' || char === '\n') break;
        
        ending = char + ending;
        
        if (vowels.includes(char)) {
            foundVowel = true;
            // Continue to capture the consonant before the vowel
            if (i > 0 && !vowels.includes(cleaned[i-1])) {
                ending = cleaned[i-1] + ending;
            }
            break;
        }
    }
    
    return foundVowel ? ending : cleaned.slice(-3);
}

/**
 * Check if two endings rhyme (Italian style).
 */
function doTheyRhyme(ending1, ending2) {
    if (!ending1 || !ending2) return false;
    
    // Normalize endings
    const e1 = ending1.toLowerCase().replace(/[àá]/g, 'a').replace(/[èé]/g, 'e').replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    const e2 = ending2.toLowerCase().replace(/[àá]/g, 'a').replace(/[èé]/g, 'e').replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    
    // Check for exact match or suffix match
    if (e1 === e2) return true;
    if (e1.endsWith(e2) || e2.endsWith(e1)) return true;
    
    // Check last 2-3 characters match
    const minLen = Math.min(e1.length, e2.length, 3);
    return e1.slice(-minLen) === e2.slice(-minLen);
}

/**
 * Get the rhyme scheme target for current verse in terza rima (ABA BCB CDC...).
 * Returns the verse index that the current verse should rhyme with, or -1 if free.
 */
function getRhymeTarget(verseIndex) {
    // Terza rima pattern:
    // Verse 0 (A): free
    // Verse 1 (B): free  
    // Verse 2 (A): rhymes with verse 0
    // Verse 3 (B): rhymes with verse 1
    // Verse 4 (C): rhymes with verse 3 (the B of previous tercet becomes A of next)
    // Verse 5 (B): rhymes with verse 4
    // etc.
    
    if (verseIndex < 2) return -1; // First two verses are free
    
    // Pattern repeats every 3 verses after the first tercet
    // For verse n >= 2:
    // If (n % 3) == 2: rhyme with n-2 (A rhymes with A)
    // If (n % 3) == 0: rhyme with n-2 (continuing chain)
    // If (n % 3) == 1: rhyme with n-2 (B rhymes with B)
    
    return verseIndex - 2;
}

/**
 * Score how well a token rhymes with the target ending.
 */
function getRhymeScore(tokenId, targetEnding) {
    if (!bpe_vocab[tokenId]) return 0;
    
    const tokenBytes = bpe_vocab[tokenId];
    const tokenText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(tokenBytes));
    const tokenEnding = getEndingSound(tokenText);
    
    if (!tokenEnding || !targetEnding) return 0;
    
    // Normalize for comparison
    const t = tokenEnding.toLowerCase().replace(/[àá]/g, 'a').replace(/[èé]/g, 'e').replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    const target = targetEnding.toLowerCase().replace(/[àá]/g, 'a').replace(/[èé]/g, 'e').replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    
    // Exact match
    if (t === target) return 1.0;
    
    // Check suffix matching (last n characters)
    for (let len = Math.min(t.length, target.length, 4); len >= 2; len--) {
        if (t.slice(-len) === target.slice(-len)) {
            return 0.5 + (len / 8); // 0.5 to 1.0 based on match length
        }
    }
    
    return 0;
}

/**
 * Check if a token ends a verse (contains newline or is followed by newline).
 */
function isVerseEndingToken(tokenId) {
    if (!bpe_vocab[tokenId]) return false;
    const tokenBytes = bpe_vocab[tokenId];
    const tokenText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(tokenBytes));
    return tokenText.includes('\n');
}

/**
 * Get the current partial verse (text since the last newline).
 */
function getCurrentPartialVerse() {
    const lines = generatedText.split('\n');
    return lines[lines.length - 1] || '';
}

/**
 * Get the last word (or partial word) from text.
 */
function getLastWord(text) {
    const words = text.trim().split(/\s+/);
    return words[words.length - 1] || '';
}

/**
 * Check if we're near end of verse (verse typically 30-50 chars in Dante's style).
 * Returns true if we should start looking for rhyming opportunities.
 */
function isNearVerseEnd() {
    const partialVerse = getCurrentPartialVerse();
    // Dante's verses are typically 10-12 syllables, roughly 30-50 characters
    return partialVerse.length >= 25;
}

/**
 * Calculate rhyme score between current partial verse ending and a target ending.
 * Considers what the verse ending would be if we add the token.
 */
function calculateRhymeScoreForToken(tokenId, targetEnding) {
    if (!bpe_vocab[tokenId] || !targetEnding) return 0;
    
    const tokenBytes = bpe_vocab[tokenId];
    const tokenText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(tokenBytes));
    
    // Get current partial verse and last word
    const partialVerse = getCurrentPartialVerse();
    const currentLastWord = getLastWord(partialVerse);
    
    // If token contains newline, check if the word before newline rhymes
    if (tokenText.includes('\n')) {
        const beforeNewline = tokenText.split('\n')[0];
        const potentialLastWord = getLastWord(currentLastWord + beforeNewline);
        const potentialEnding = getEndingSound(potentialLastWord);
        
        if (potentialEnding && doTheyRhyme(potentialEnding, targetEnding)) {
            return 1.0; // Perfect match - ends verse with rhyme
        }
        return 0;
    }
    
    // If token doesn't contain newline, check if it could form a rhyming word
    const potentialWord = currentLastWord + tokenText;
    const potentialEnding = getEndingSound(potentialWord);
    
    if (potentialEnding && doTheyRhyme(potentialEnding, targetEnding)) {
        return 0.5; // Partial match - building towards rhyme
    }
    
    return 0;
}
