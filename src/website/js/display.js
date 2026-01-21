/**
 * display.js
 * ==========
 * Text display, UI updates, and generation controls.
 */

// ============================================================================
// Text Display & Canto Management
// ============================================================================

/**
 * Create a canto separator element.
 */
function createCantoSeparator(cantoNum) {
    const separator = document.createElement('div');
    separator.className = 'canto-separator';
    separator.textContent = `CANTO ${toRoman(cantoNum)}`;
    return separator;
}

/**
 * Convert number to Roman numerals.
 */
function toRoman(num) {
    const romanNumerals = [
        ['M', 1000], ['CM', 900], ['D', 500], ['CD', 400],
        ['C', 100], ['XC', 90], ['L', 50], ['XL', 40],
        ['X', 10], ['IX', 9], ['V', 5], ['IV', 4], ['I', 1]
    ];
    let result = '';
    for (const [letter, value] of romanNumerals) {
        while (num >= value) {
            result += letter;
            num -= value;
        }
    }
    return result;
}

/**
 * Update the display with new text.
 */
function updateDisplay(newChar) {
    generatedText += newChar;
    charsSinceLastCanto += newChar.length;

    // Check for canto break
    if (charsSinceLastCanto >= CHARS_PER_CANTO && (newChar === '.' || newChar === '\n')) {
        // Insert canto break
        cantoCount++;
        charsSinceLastCanto = 0;

        // Add separator to DOM
        const separator = createCantoSeparator(cantoCount);
        textContainer.insertBefore(separator, cursorEl);
    }

    // Update text content (without highlighting for now, highlighting is done in updateInlineContextHighlight)
    if (!showContextWindow) {
        textOutput.textContent = generatedText;
    }

    // Smart auto-scroll: only if user is near the bottom (within 100px)
    const scrollBottom = textContainer.scrollHeight - textContainer.scrollTop - textContainer.clientHeight;
    if (scrollBottom < 100) {
        textContainer.scrollTop = textContainer.scrollHeight;
    }
}

/**
 * Update inline context highlighting in main text display.
 * Shows the portion of text within the context window with blue highlighting.
 */
function updateInlineContextHighlight(tokens) {
    if (!showContextWindow) return;
    
    // Get context tokens
    const contextTokens = tokens.slice(-effectiveBlockSize);
    
    // Calculate how many characters are in context vs outside
    // We need to decode the full token history and the context portion
    const fullBytes = [];
    for (const t of tokens) {
        if (bpe_vocab[t]) {
            fullBytes.push(...bpe_vocab[t]);
        }
    }
    const fullText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(fullBytes));
    
    const contextBytes = [];
    for (const t of contextTokens) {
        if (bpe_vocab[t]) {
            contextBytes.push(...bpe_vocab[t]);
        }
    }
    const contextText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(contextBytes));
    
    // Split the generatedText into non-context and context portions
    // The context portion is the last N characters that correspond to context tokens
    const contextLength = contextText.length;
    const totalLength = generatedText.length;
    
    if (contextLength >= totalLength) {
        // All text is in context
        textOutput.innerHTML = `<span class="context-highlight">${escapeHtml(generatedText)}</span>`;
    } else {
        // Split into non-context and context parts
        const nonContextPart = generatedText.slice(0, totalLength - contextLength);
        const contextPart = generatedText.slice(totalLength - contextLength);
        textOutput.innerHTML = escapeHtml(nonContextPart) + 
            `<span class="context-highlight">${escapeHtml(contextPart)}</span>`;
    }
}

/**
 * Render the full text with canto breaks.
 */
function renderFullText() {
    // Clear existing separators
    textContainer.querySelectorAll('.canto-separator').forEach(el => el.remove());
    textOutput.textContent = generatedText;
}

/**
 * Update the context window display with current tokens.
 */
function updateContextWindowDisplay(tokens, newTokenCount = 1) {
    if (!showContextWindow || !ctxContent) return;

    // Get the context window tokens (last effectiveBlockSize tokens)
    const contextTokens = tokens.slice(-effectiveBlockSize);
    
    // Decode all tokens to get the text
    const allBytes = [];
    for (const t of contextTokens) {
        if (bpe_vocab[t]) {
            allBytes.push(...bpe_vocab[t]);
        }
    }
    
    // Decode to text
    const contextText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(allBytes));
    
    // Calculate where the new tokens start in the text
    const oldTokens = contextTokens.slice(0, -newTokenCount);
    const oldBytes = [];
    for (const t of oldTokens) {
        if (bpe_vocab[t]) {
            oldBytes.push(...bpe_vocab[t]);
        }
    }
    const oldText = new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(oldBytes));
    
    // Split text into old and new parts
    const oldPart = contextText.slice(0, oldText.length);
    const newPart = contextText.slice(oldText.length);
    
    // Update display with highlighted new token
    ctxContent.innerHTML = escapeHtml(oldPart) + 
        (newPart ? `<span class="ctx-new">${escapeHtml(newPart)}</span>` : '');
    
    // Update token count
    ctxTokenCount.textContent = `${contextTokens.length} / ${effectiveBlockSize} tokens`;
    
    // Auto-scroll to bottom
    contextWindowDisplay.scrollTop = contextWindowDisplay.scrollHeight;
}

/**
 * Escape HTML characters for safe display.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Interactive Editing
// ============================================================================

/**
 * Enable text editing mode.
 */
function enableEditing() {
    textOutput.contentEditable = 'true';
    textOutput.classList.add('editable');
    editIndicator.classList.add('visible');
    cursorEl.style.display = 'none';
}

/**
 * Disable text editing mode.
 */
function disableEditing() {
    textOutput.contentEditable = 'false';
    textOutput.classList.remove('editable');
    editIndicator.classList.remove('visible');
    cursorEl.style.display = 'inline-block';

    // Sync text content back
    generatedText = textOutput.textContent;
}

// ============================================================================
// Generation Loop
// ============================================================================

async function startGeneration() {
    if (isGenerating) return;

    isGenerating = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = 'GENERATING...';
    statusEl.classList.add('active');

    disableEditing();

    // Initialize with seed text if empty
    if (generatedText.length === 0) {
        generatedText = 'Nel mezzo del cammin di nostra vita ';
        textOutput.textContent = generatedText;
        charsSinceLastCanto = generatedText.length;
    }

    // Convert current text to tokens
    let tokens = encode(generatedText);
    if (tokens.length === 0) tokens = [0];

    // Generation loop
    while (isGenerating) {
        try {
            const nextToken = await generateNext(tokens);
            tokens.push(nextToken);

            if (tokens.length > effectiveBlockSize) {
                tokens = tokens.slice(-effectiveBlockSize);
            }

            const char = decode([nextToken]);
            
            // Track verse endings for Dante rhyme mode
            if (danteRhymeMode && char.includes('\n')) {
                // Extract the ending of the verse that just ended
                // generatedText still contains the text BEFORE this newline
                const lines = generatedText.split('\n');
                // Get the last non-empty line (the verse that just ended)
                const lastVerse = lines.filter(l => l.trim().length > 0).pop() || '';
                const ending = getEndingSound(lastVerse);
                console.log(`Verse ${currentVerseNumber}: "${lastVerse.slice(-30)}" -> ending: "${ending}"`);
                verseEndings.push(ending);
                currentVerseNumber++;
            }
            
            updateDisplay(char);
            updateContextWindowDisplay(tokens);
            updateInlineContextHighlight(tokens);

            await new Promise(resolve => setTimeout(resolve, speed));

        } catch (error) {
            console.error('Generation error:', error);
            statusEl.textContent = 'ERROR: ' + error.message;
            stopGeneration();
            break;
        }
    }
}

function stopGeneration() {
    isGenerating = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'PAUSED — CLICK TEXT TO EDIT';
    statusEl.classList.remove('active');

    enableEditing();
}

function clearText() {
    stopGeneration();
    generatedText = '';
    cantoCount = 1;
    charsSinceLastCanto = 0;
    textOutput.textContent = '';
    textContainer.querySelectorAll('.canto-separator').forEach(el => el.remove());
    statusEl.textContent = 'CLEARED — PRESS START';
    disableEditing();
    resetDecoder(); // Clear pending UTF-8 bytes
    // Reset rhyme tracking
    verseEndings = [];
    currentVerseNumber = 0;
}
