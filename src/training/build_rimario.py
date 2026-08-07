"""
Build Rimario (Italian Rhyme Dictionary)
==========================================
Generates a JSON rhyme dictionary grouped by stressed suffix.

Sources (in priority order):
1. Training corpus files (data/clean/pretrain.txt, finetune.txt)
2. A curated seed list of high-frequency Italian poetic words

For each word the script identifies the stressed suffix:
- Explicit accent mark → stress is on that vowel (tronca: città → à)
- Otherwise assume paroxytone (piana): stress on penultimate vowel
  (covers ~80 % of Italian words, good enough for poetry)

Output: model/rimario.json
  {
    "<suffix>": ["word1", "word2", ...],
    ...
  }
"""

import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
MODEL_DIR = os.path.join(BASE_DIR, "model")

VOWELS = set("aeiouàèéìíòóùú")
ACCENTED_MAP = {
    "à": "a", "è": "e", "é": "e", "ì": "i", "í": "i",
    "ò": "o", "ó": "o", "ù": "u", "ú": "u",
}
ACCENTED_VOWELS = set(ACCENTED_MAP.keys())

# Common Italian diphthongs (two vowels pronounced as one syllable)
DIPHTHONGS = {
    "ia", "ie", "io", "iu",
    "ua", "ue", "ui", "uo",
    "ai", "ei", "oi", "au", "eu",
}

# Minimum word length to include in rimario
MIN_WORD_LENGTH = 3
# Minimum words sharing a suffix to keep that suffix group
MIN_GROUP_SIZE = 2
# Maximum words per suffix group (keep most frequent)
MAX_GROUP_SIZE = 200


def normalize_word(word: str) -> str:
    """Lowercase and strip non-alphabetic edges, keep accents."""
    return word.lower().strip().strip(".,;:!?\"'()[]{}«»—–-…")


def _split_syllable_nuclei(word: str) -> list:
    """
    Return a list of (start_pos, end_pos) for each syllable nucleus in the word.
    
    A nucleus is a vowel or diphthong. Two adjacent vowels forming a diphthong
    (e.g. 'io', 'uo') count as ONE nucleus; a hiatus ('ìa', accented + vowel)
    counts as TWO.
    
    This lets us find the penultimate *syllable* vowel, not just the
    penultimate character vowel — fixing e.g. "occhio" whose last two
    vowels 'i'+'o' are one diphthong, not two syllables.
    """
    nuclei = []
    i = 0
    plain = ""
    for c in word:
        plain += ACCENTED_MAP.get(c, c)
    
    while i < len(plain):
        if plain[i] in "aeiou":
            start = i
            # Absorb diphthong: if next char is also a vowel and the pair is
            # a known diphthong (and neither vowel carries an explicit accent
            # that would signal hiatus), merge them into one nucleus.
            while (i + 1 < len(plain) and plain[i + 1] in "aeiou"
                   and plain[i:i+2] in DIPHTHONGS
                   and word[i] not in ACCENTED_VOWELS
                   and word[i + 1] not in ACCENTED_VOWELS):
                i += 1
            nuclei.append((start, i))
            i += 1
        else:
            i += 1
    return nuclei


def get_stressed_suffix(word: str) -> Optional[str]:
    """
    Extract the rhyme suffix starting from the stressed vowel.

    Rules:
    1. If the word has an accented vowel, stress is on that vowel.
       The suffix runs from that vowel to the end of the word.
    2. Otherwise, assume paroxytone (stress on penultimate *syllable*).
       Diphthongs like 'io' in "occhio" count as one syllable, so
       the penultimate syllable nucleus — not the penultimate vowel
       character — determines where the suffix starts.
    3. For monosyllables or words with only one nucleus, return the
       whole word from its only vowel onward.
    """
    word = normalize_word(word)
    if not word or len(word) < MIN_WORD_LENGTH:
        return None

    # Strip trailing non-alpha (punctuation that survived normalize)
    word = re.sub(r"[^a-zàèéìíòóùú]", "", word)
    if not word or len(word) < MIN_WORD_LENGTH:
        return None

    # 1. Look for explicit accent
    for i, ch in enumerate(word):
        if ch in ACCENTED_VOWELS:
            # Suffix from this vowel onward, with accent normalized
            suffix = ""
            for c in word[i:]:
                suffix += ACCENTED_MAP.get(c, c)
            return suffix if len(suffix) >= 2 else None

    # 2. No explicit accent → paroxytone (penultimate syllable nucleus)
    nuclei = _split_syllable_nuclei(word)

    if not nuclei:
        return None

    if len(nuclei) >= 2:
        # Penultimate syllable nucleus
        stress_start = nuclei[-2][0]
    else:
        # Only one nucleus
        stress_start = nuclei[0][0]

    suffix = word[stress_start:]
    return suffix if len(suffix) >= 2 else None


def extract_words_from_corpus() -> dict:
    """Extract words and their frequencies from the training corpus."""
    word_freq = defaultdict(int)

    for filename in ["pretrain.txt", "finetune.txt"]:
        filepath = os.path.join(DATA_CLEAN_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  Corpus file not found: {filepath}")
            continue

        print(f"  Reading {filepath}...")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Extract words
        words = re.findall(r"[a-zA-ZàèéìíòóùúÀÈÉÌÍÒÓÙÚ']+", text.lower())
        for w in words:
            # Remove apostrophe-only entries and very short words
            w = w.strip("'")
            if len(w) >= MIN_WORD_LENGTH:
                word_freq[w] += 1

        print(f"    Found {len(words):,} word tokens")

    return word_freq


# Curated seed list of common Italian poetic words, organized by category.
# This ensures the rimario has good coverage even without the full corpus.
SEED_WORDS = [
    # Dante's Commedia - high frequency
    "vita", "via", "gente", "mente", "parte", "morte", "luce", "voce",
    "mondo", "fondo", "secondo", "giocondo", "profondo", "tondo", "biondo",
    "terra", "guerra", "serra", "afferra", "atterra",
    "cielo", "velo", "gelo", "pelo", "stelo", "angelo", "modello",
    "amore", "dolore", "cuore", "errore", "fiore", "signore", "valore",
    "onore", "splendore", "ardore", "furore", "timore", "colore", "calore",
    "oscura", "paura", "natura", "misura", "figura", "ventura", "altura",
    "dura", "pura", "sicura", "matura", "futura", "avventura", "struttura",
    "smarrita", "ardita", "salita", "ferita", "unita", "gradita", "spedita",
    "infinita", "finita", "partita", "rapita", "scolpita",
    "forte", "sorte", "morte", "corte", "porte", "consorte", "conforte",
    "notte", "lotte", "grotte", "dotte", "rotte",
    "stelle", "belle", "pelle", "quelle", "sorelle", "novelle", "favelle",
    "cosa", "rosa", "sposa", "pietosa", "nascosa", "dolorosa", "gloriosa",
    "alma", "calma", "palma", "salma",
    "bene", "viene", "tiene", "conviene", "sostiene", "contiene", "mantiene",
    "vista", "trista", "conquista", "artista", "lista",
    "canto", "tanto", "santo", "pianto", "manto", "vanto", "incanto",
    "vento", "momento", "tormento", "sentimento", "lamento", "contento",
    "pensiero", "sentiero", "cavaliero", "altiero", "intiero", "guerriero",
    "parola", "sola", "vola", "scuola", "consola", "aiuola",
    "pace", "face", "piace", "giace", "audace", "verace", "capace", "tenace",
    "sera", "vera", "nera", "leggera", "severa", "primavera", "maniera",
    "anno", "affanno", "danno", "inganno", "tiranno",
    "arte", "parte", "carte", "marte", "sparte",
    "occhio", "ginocchio", "orecchio", "vecchio", "specchio",
    "acqua", "lingua", "sangue",
    "padre", "madre", "leggiadre",
    "fuoco", "gioco", "loco", "poco",
    "alto", "salto", "assalto", "smalto",
    "ombra", "sgombra", "ingombra",
    "tempo", "esempio", "sempre",
    "grande", "domande", "ghirlande",
    "giusto", "augusto", "robusto", "gusto", "ingiusto",
    "mano", "piano", "umano", "sovrano", "lontano", "invano", "arcano",
    "primo", "ultimo", "animo", "intimo",
    "porto", "morto", "conforto", "torto", "accorto", "corto", "risorto",
    "fiamma", "dramma", "programma",
    "legge", "regge", "protegge", "corregge",
    "stato", "prato", "amato", "beato", "passato", "nato",
    "raggio", "viaggio", "coraggio", "saggio", "aggio", "passaggio",
    "petto", "detto", "affetto", "aspetto", "concetto", "diletto", "effetto",
    "ingegno", "segno", "degno", "regno", "sdegno", "disegno",
    "passo", "basso", "lasso", "grasso",
    "spera", "schiera", "bandiera", "preghiera", "riviera", "costiera",
    "desio", "rio", "oblio", "natio",
    "trono", "dono", "suono", "buono", "tuono", "perdono",
    "ira", "mira", "gira", "sospira", "ammira", "aspira", "ispira",
    "braccio", "ghiaccio", "palazzo", "spazio", "laccio",
    "volto", "molto", "sepolto", "ascolto", "sciolto", "raccolto",
    "forma", "norma", "riforma",
    "chiaro", "amaro", "raro", "caro", "avaro", "imparo",
    "ora", "ancora", "dimora", "aurora", "adora", "lavora", "divora",
    "speranza", "danza", "usanza", "possanza", "lontananza", "sembianza",
    "mente", "gente", "presente", "ardente", "corrente", "fremente",
    "potente", "vincente", "possente", "dolente", "silente", "prudente",
    "acceso", "peso", "inteso", "teso", "difeso", "offeso", "sospeso",
    "vita", "ita", "udita", "punita", "fornita", "nutrita", "vestita",
    "grazia", "audacia", "minaccia",
    "profeta", "poeta", "meta", "quieta", "lieta", "segreta",
    "eterno", "interno", "inferno", "inverno", "materno", "governo",
    "giorno", "ritorno", "intorno", "soggiorno", "contorno", "adorno",
    "atto", "fatto", "tratto", "ratto", "esatto", "contratto",
    "sogno", "bisogno", "vergogno",
    "pena", "serena", "piena", "catena", "scena", "arena",
    "speme", "teme", "geme", "insieme", "preme", "freme",
    "core", "signore", "pastore", "dottore", "scrittore", "autore",
    "terra", "guerra", "afferra", "serra",
    "campo", "lampo", "scampo", "vampo",
    "punto", "giunto", "assunto", "defunto", "congiunto",
    "spirto", "dritto", "scritto", "afflitto", "conflitto", "editto",
    "uomo", "duomo", "pomo",
    "causa", "pausa", "plausa",
    "grido", "nido", "lido", "fido",
    "onda", "sponda", "seconda", "gioconda", "profonda",
    "gloria", "storia", "memoria", "vittoria",
    "gioia", "noia",
]


def build_rimario():
    """Build the rimario and save to model/rimario.json."""
    print("=== Building Rimario (Italian Rhyme Dictionary) ===\n")

    # 1. Collect words from corpus
    print("Phase 1: Extracting words from corpus...")
    word_freq = extract_words_from_corpus()
    print(f"  Corpus words: {len(word_freq):,}\n")

    # 2. Add seed words (with low frequency so corpus words win ties)
    print("Phase 2: Adding seed words...")
    for w in SEED_WORDS:
        w = normalize_word(w)
        if w not in word_freq:
            word_freq[w] = 1  # Seed words get minimum frequency
    print(f"  Total unique words: {len(word_freq):,}\n")

    # 3. Group by stressed suffix
    print("Phase 3: Grouping by stressed suffix...")
    suffix_groups = defaultdict(list)
    skipped = 0

    for word, freq in word_freq.items():
        suffix = get_stressed_suffix(word)
        if suffix is None:
            skipped += 1
            continue
        suffix_groups[suffix].append((word, freq))

    print(f"  Suffixes found: {len(suffix_groups):,}")
    print(f"  Words skipped (too short/no vowel): {skipped:,}\n")

    # 4. Filter and sort
    print("Phase 4: Filtering and sorting...")
    rimario = {}
    total_words = 0

    for suffix in sorted(suffix_groups.keys()):
        entries = suffix_groups[suffix]

        # Filter: need at least MIN_GROUP_SIZE words for rhyming
        if len(entries) < MIN_GROUP_SIZE:
            continue

        # Sort by frequency (descending), keep top MAX_GROUP_SIZE
        entries.sort(key=lambda x: -x[1])
        words = [w for w, _ in entries[:MAX_GROUP_SIZE]]

        # Remove exact duplicates
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        words = unique_words

        if len(words) >= MIN_GROUP_SIZE:
            rimario[suffix] = words
            total_words += len(words)

    print(f"  Suffix groups: {len(rimario):,}")
    print(f"  Total words: {total_words:,}\n")

    # 5. Show top groups
    print("Top 20 rhyme groups by size:")
    top_groups = sorted(rimario.items(), key=lambda x: -len(x[1]))[:20]
    for suffix, words in top_groups:
        preview = ", ".join(words[:6])
        if len(words) > 6:
            preview += f", ... (+{len(words) - 6})"
        print(f"  -{suffix}: {len(words)} words — {preview}")

    # 6. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, "rimario.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rimario, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    file_size = os.path.getsize(output_path)
    print(f"\nSaved rimario to {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    # Sanity checks
    print("\n=== Sanity Checks ===")
    test_pairs = [
        ("vita", "smarrita"),     # Perfect rhyme on -ita
        ("oscura", "paura"),      # Not a strict rhyme (ura vs aura)
        ("canto", "vento"),       # NOT a rhyme (old bug: last 2 letters match)
        ("amore", "dolore"),      # Perfect rhyme on -ore
        ("stelle", "belle"),      # Perfect rhyme on -elle
    ]
    for w1, w2 in test_pairs:
        s1 = get_stressed_suffix(w1)
        s2 = get_stressed_suffix(w2)
        match = s1 == s2
        symbol = "[OK] RHYME" if match else "[--] no rhyme"
        print(f"  {w1}(-{s1}) vs {w2}(-{s2}): {symbol}")

    # Verify the old false positive is fixed
    assert get_stressed_suffix("canto") != get_stressed_suffix("vento"), \
        "BUG: canto/vento should NOT rhyme!"
    assert get_stressed_suffix("vita") == get_stressed_suffix("smarrita"), \
        "BUG: vita/smarrita SHOULD rhyme!"
    # Verify diphthong-aware syllable splitting
    assert get_stressed_suffix("occhio") != get_stressed_suffix("desio"), \
        "BUG: occhio/desio should NOT rhyme (different stressed syllable)!"
    assert get_stressed_suffix("occhio") == get_stressed_suffix("ginocchio"), \
        "BUG: occhio/ginocchio SHOULD rhyme on -occhio!"
    assert get_stressed_suffix("occhio") != get_stressed_suffix("specchio"), \
        "occhio (-occhio) and specchio (-ecchio) have different stressed vowels"
    print("\n[OK] All sanity checks passed")

    return rimario


if __name__ == "__main__":
    build_rimario()
