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
 * For Italian poetry, rhymes are based on the ending from the last stressed vowel.
 * Examples: "vita" -> "ita", "oscura" -> "ura", "selva" -> "elva"
 */
function getEndingSound(text) {
    // Clean: remove punctuation and normalize
    let cleaned = text.toLowerCase().trim();
    // Remove trailing punctuation
    cleaned = cleaned.replace(/[.,;:!?'"»«\-–—]+$/g, '');
    if (cleaned.length === 0) return '';
    
    // Get the last word
    const words = cleaned.split(/\s+/);
    const lastWord = words[words.length - 1] || '';
    if (lastWord.length === 0) return '';
    
    // Italian vowels (including accented)
    const vowels = 'aeiouàèéìòóù';
    
    // For Italian rhymes, we want the ending from the last stressed syllable
    // In most cases, this is the last 2-4 characters starting from a vowel
    // Strategy: find the second-to-last vowel position, or use last 3-4 chars
    
    // Find positions of all vowels in the word
    const vowelPositions = [];
    for (let i = 0; i < lastWord.length; i++) {
        if (vowels.includes(lastWord[i])) {
            vowelPositions.push(i);
        }
    }
    
    if (vowelPositions.length === 0) {
        // No vowels, just return last 3 chars
        return lastWord.slice(-3);
    }
    
    // For Italian rhymes, typically we want from the second-to-last vowel
    // "vita" (i at 1, a at 3) -> from position 1 = "ita"
    // "oscura" (o at 0, u at 3, a at 5) -> from position 3 = "ura"  
    // "selva" (e at 1, a at 4) -> from position 1 = "elva"
    
    let startPos;
    if (vowelPositions.length >= 2) {
        // Start from the second-to-last vowel
        startPos = vowelPositions[vowelPositions.length - 2];
    } else {
        // Only one vowel, start from it
        startPos = vowelPositions[0];
    }
    
    // Extract the ending
    const ending = lastWord.slice(startPos);
    
    // Normalize accented vowels
    return ending
        .replace(/[àá]/g, 'a')
        .replace(/[èé]/g, 'e')
        .replace(/[ìí]/g, 'i')
        .replace(/[òó]/g, 'o')
        .replace(/[ùú]/g, 'u');
}

/**
 * Check if two endings rhyme (Italian style).
 * Requires at least 2 characters to match at the end.
 */
function doTheyRhyme(ending1, ending2) {
    if (!ending1 || !ending2) return false;
    
    // Both endings must be at least 2 characters for a valid rhyme
    if (ending1.length < 2 || ending2.length < 2) return false;
    
    // Normalize endings
    const e1 = ending1.toLowerCase().replace(/[àá]/g, 'a').replace(/[èé]/g, 'e').replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    const e2 = ending2.toLowerCase().replace(/[àá]/g, 'a').replace(/[èé]/g, 'e').replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    
    // Check for exact match
    if (e1 === e2) return true;
    
    // Check suffix match (only if suffix is at least 2 chars)
    if (e1.length >= 2 && e2.endsWith(e1)) return true;
    if (e2.length >= 2 && e1.endsWith(e2)) return true;
    
    // Check last 2-3 characters match (minimum 2 required)
    const matchLen = Math.min(e1.length, e2.length, 3);
    if (matchLen >= 2) {
        return e1.slice(-matchLen) === e2.slice(-matchLen);
    }
    
    return false;
}

/**
 * Find all tokens in vocabulary that would create a rhyme with target.
 * Returns array of {tokenId, text, ending} sorted by how well they match.
 */
function findRhymingTokens(targetEnding, probs) {
    if (!targetEnding || !bpe_vocab) return [];
    
    const rhymingTokens = [];
    
    // Search through all tokens in vocabulary
    for (let tokenId = 0; tokenId < Object.keys(bpe_vocab).length; tokenId++) {
        if (!bpe_vocab[tokenId]) continue;
        
        const tokenBytes = bpe_vocab[tokenId];
        const tokenText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(tokenBytes));
        
        // Clean the token text to check for rhyme
        // We want tokens that could be the END of a word
        // So we remove trailing punctuation/spaces for the check
        const cleanText = tokenText.trim().replace(/[.,;:!?'"»«\-–—]+$/g, '');
        
        // Skip short tokens or tokens that are just punctuation
        if (cleanText.length < 2) continue;
        
        // Get the ending of the token
        const tokenEnding = getEndingSound(cleanText);
        if (!tokenEnding || tokenEnding.length < 2) continue;
        
        // Check if it rhymes
        if (doTheyRhyme(tokenEnding, targetEnding)) {
            const prob = probs ? probs[tokenId] : 0;
            rhymingTokens.push({
                tokenId,
                text: tokenText, // Keep original text
                cleanText: cleanText,
                ending: tokenEnding,
                prob
            });
        }
    }
    
    // Sort by probability (highest first)
    rhymingTokens.sort((a, b) => b.prob - a.prob);
    
    return rhymingTokens;
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
    
    // Clean the token text
    const cleanToken = tokenText.toLowerCase()
        .replace(/[àá]/g, 'a').replace(/[èé]/g, 'e')
        .replace(/[ìí]/g, 'i').replace(/[òó]/g, 'o').replace(/[ùú]/g, 'u');
    
    // Get current partial verse and last word
    const partialVerse = getCurrentPartialVerse();
    const currentLastWord = getLastWord(partialVerse);
    
    // If token contains newline, check if the word before newline rhymes
    if (tokenText.includes('\n')) {
        const beforeNewline = tokenText.split('\n')[0];
        const potentialLastWord = currentLastWord + beforeNewline;
        const potentialEnding = getEndingSound(potentialLastWord);
        
        if (potentialEnding && doTheyRhyme(potentialEnding, targetEnding)) {
            return 1.0; // Perfect match - ends verse with rhyme
        }
        // Even if newline token doesn't rhyme perfectly, check partial match
        if (potentialEnding && targetEnding.endsWith(potentialEnding.slice(-2))) {
            return 0.3; // Weak match
        }
        return 0;
    }
    
    // If token doesn't contain newline, check if it could form a rhyming word
    const potentialWord = currentLastWord + tokenText;
    const potentialEnding = getEndingSound(potentialWord);
    
    // Check for rhyme match
    if (potentialEnding && doTheyRhyme(potentialEnding, targetEnding)) {
        return 0.7; // Good match - building towards rhyme
    }
    
    // Check if the token itself contains the rhyme pattern (could lead to rhyme)
    // e.g., if targetEnding is "ita", and token contains "it", that's promising
    const target2 = targetEnding.slice(-2);
    const target3 = targetEnding.slice(-3);
    
    if (cleanToken.includes(target3)) {
        return 0.6; // Token contains the rhyme pattern
    }
    if (cleanToken.includes(target2)) {
        return 0.4; // Token contains part of the rhyme pattern
    }
    if (cleanToken.endsWith(target2.charAt(0))) {
        return 0.2; // Token ends with start of rhyme
    }
    
    return 0;
}
