"""
Build Rimario (Italian Rhyme Dictionary)
==========================================
Generates a JSON rhyme dictionary grouped by stressed suffix.

Sources (in priority order):
1. Complete Divina Commedia text (Inferno, Purgatorio, Paradiso)
2. Training corpus files (data/clean/pretrain.txt, finetune.txt) if present
3. Curated seed list of high-frequency Italian poetic words

Identifies stressed suffixes using phonological rules:
- Explicit accents (città -> a, partì -> i)
- Oxytone endings (-ai, -ei, -ui, -oi: trovai -> ai, fei -> ei)
- Hiatus exceptions (pa-ù-ra -> ura, de-sì-o -> io, o-blì-o -> io, v-ì-a -> ia)
- Diacritic 'i' handling after c/g before a/o/u (giunto -> unto, gioco -> oco)
- Penultimate syllable nucleus for paroxytones (covers ~85% of Italian vocabulary)

Output: model/rimario.json
  {
    "<suffix>": ["word1", "word2", ...],
    ...
  }
"""

import json
import os
import re
import urllib.request
from collections import defaultdict
from typing import Optional, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
MODEL_DIR = os.path.join(BASE_DIR, "model")

VOWELS = set("aeiouàèéìíòóùúAEIOUÀÈÉÌÍÒÓÙÚ")
ACCENTED_MAP = {
    "à": "a", "è": "e", "é": "e", "ì": "i", "í": "i",
    "ò": "o", "ó": "o", "ù": "u", "ú": "u",
}
ACCENTED_VOWELS = set(ACCENTED_MAP.keys())

# Common Italian diphthongs
DIPHTHONGS = {
    "ia", "ie", "io", "iu",
    "ua", "ue", "ui", "uo",
    "ai", "ei", "oi", "au", "eu",
}

# Explicit known hiatus words or stems where stress is on the hiatus vowel
KNOWN_HIATUS_STEMS = {
    "paura": "ura",
    "paure": "ure",
}

# Words ending in tonic -io (de-sì-o, o-blì-o, r-ì-o, na-tì-o, add-ì-o)
TONIC_IO_WORDS = {
    "desio", "oblio", "rio", "natio", "mormorio", "gentilio", "pendio", "addio", 
    "fio", "disio", "campio", "brischio", "restio", "brusio", "ronzio", "calpestio"
}

# Words ending in tonic -ia (v-ì-a, pr-ì-a, m-ì-a, s-ì-a, com-pa-gn-ì-a, fol-l-ì-a, ecc.)
TONIC_IA_WORDS = {
    "via", "pria", "mia", "sia", "follia", "armonia", "poesia", "cortesia",
    "allegria", "compagnia", "balia", "ria", "magia", "fantasia", "ironia",
    "bugia", "corsia", "fabbria", "malia", "villania", "gelosia", "fiancheria",
    "fantesia", "baratteria", "idropesia", "epia", "abbadia", "dia", "disvia",
    "invia", "devia", "godia", "udia", "sentia", "partia", "uscia", "salia",
    "smarria", "maria", "girolomia", "tenia", "vedia", "faria", "daria", "saria",
    "avria", "poria", "credea", "solia", "arderia", "valia", "pazia"
}

NON_TONIC_IA_WORDS = {
    "grazia", "audacia", "minaccia", "angoscia", "striscia", "lascia", "faccia",
    "provincia", "materia", "memoria", "gloria", "storia", "vittoria", "notizia",
    "sentenzia", "ingiuria", "ignominia", "calunnia", "curia", "furia"
}

MIN_WORD_LENGTH = 2
MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 250


def normalize_word(word: str) -> str:
    """Lowercase and strip non-alphabetic edges, keep accents."""
    return word.lower().strip().strip(".,;:!?\"'()[]{}«»—–-…`´’‘“ ”")


def get_stressed_suffix(word: str) -> Optional[str]:
    """
    Extract the rhyme suffix starting from the stressed vowel.
    
    1. Look for explicit accents.
    2. Check known poetic hiatus words (paura -> ura, desio -> io, via -> ia).
    3. Check oxytone diphthongs (-ai, -ei, -ui, -oi: trovai -> ai).
    4. Handle diacritic 'i' after c/g before a/o/u (giunto -> unto).
    5. Split into syllable nuclei and pick penultimate for paroxytones.
    """
    w = normalize_word(word)
    if not w:
        return None
    w = re.sub(r"[^a-zàèéìíòóùú]", "", w)
    if not w or len(w) < MIN_WORD_LENGTH:
        return None

    # 1. Check known explicit stems / whole words
    if w in KNOWN_HIATUS_STEMS:
        return KNOWN_HIATUS_STEMS[w]
    if w in TONIC_IO_WORDS:
        return "io"
    if w in TONIC_IA_WORDS or (w.endswith("ia") and len(w) <= 4):
        return "ia"
    # Imperfect / conditional archaic endings in -ia (smarria, sentia, moria, etc.)
    if w.endswith("ia") and len(w) >= 4:
        if w not in NON_TONIC_IA_WORDS and (w.endswith(("ria", "dia", "tia", "via", "nia", "lia")) or w in TONIC_IA_WORDS):
            return "ia"

    # 2. Check explicit accent mark
    for i, ch in enumerate(w):
        if ch in ACCENTED_VOWELS:
            suffix = "".join(ACCENTED_MAP.get(c, c) for c in w[i:])
            return suffix if len(suffix) >= 1 else None

    # 3. Oxytone (tronca) verbal / common endings: -ai, -ei, -ui, -oi
    if len(w) >= 3 and w.endswith(("ai", "ei", "ui", "oi")):
        return w[-2:]

    # 4. Hiatus pattern: vowel + u + consonant + a (like paura -> pa-u-ra)
    if "aura" in w and w.endswith("aura") and w != "aura" and not w.endswith("laura"):
        return "ura"

    # 5. Handle diacritic 'i' after 'c' or 'g' before 'a', 'o', 'u'
    plain = "".join(ACCENTED_MAP.get(c, c) for c in w)
    nuclei = []
    i = 0
    while i < len(plain):
        if plain[i] in "aeiou":
            # Check if this 'i' is diacritic after c/g followed by a/o/u
            if plain[i] == 'i' and i > 0 and plain[i-1] in {'c', 'g'} and i + 1 < len(plain) and plain[i+1] in {'a', 'o', 'u'}:
                i += 1
                continue

            start = i
            # Check diphthong
            while (i + 1 < len(plain) and plain[i + 1] in "aeiou" 
                   and plain[i:i+2] in DIPHTHONGS):
                # Avoid merging known hiatus pairs
                if plain[i:i+2] in {"ia", "io", "ie", "ea", "eo", "oa"}:
                    if w in TONIC_IO_WORDS or w in TONIC_IA_WORDS:
                        break
                i += 1
            nuclei.append((start, i))
            i += 1
        else:
            i += 1

    if not nuclei:
        return plain[-3:] if len(plain) >= 3 else plain

    if len(nuclei) >= 2:
        stress_start = nuclei[-2][0]
    else:
        stress_start = nuclei[0][0]

    suffix = plain[stress_start:]
    return suffix if len(suffix) >= 2 else None


def fetch_divina_commedia() -> str:
    """Download and return the combined text of Inferno, Purgatorio, and Paradiso."""
    urls = [
        ("Inferno", "https://raw.githubusercontent.com/asperti/Dante/master/inferno.txt"),
        ("Purgatorio", "https://raw.githubusercontent.com/asperti/Dante/master/purgatorio.txt"),
        ("Paradiso", "https://raw.githubusercontent.com/asperti/Dante/master/paradiso.txt"),
    ]
    combined_texts = []
    
    for name, url in urls:
        print(f"Fetching {name} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as res:
                text = res.read().decode("utf-8", errors="ignore")
                print(f"  -> Successfully fetched {name} ({len(text):,} characters)")
                combined_texts.append(text)
        except Exception as e:
            print(f"  -> Warning: failed to download {name}: {e}")
            
    return "\n\n".join(combined_texts)


def extract_words_from_corpus() -> Dict[str, int]:
    """Extract words and frequencies from Commedia and training corpus."""
    word_freq = defaultdict(int)

    # 1. Download or load Commedia
    commedia_text = fetch_divina_commedia()
    if commedia_text:
        # Give Commedia words high priority (frequency weight)
        for line in commedia_text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "Inferno", "Purgatorio", "Paradiso", "Title:", "Description:", "Source:")):
                continue
            words = re.findall(r"[a-zA-ZàèéìíòóùúÀÈÉÌÍÒÓÙÚ']+", line)
            if not words:
                continue
            # Give rhyme-position words extra weight
            last_word = normalize_word(words[-1].strip("'"))
            if len(last_word) >= MIN_WORD_LENGTH:
                word_freq[last_word] += 10
            for w in words:
                w_norm = normalize_word(w.strip("'"))
                if len(w_norm) >= MIN_WORD_LENGTH:
                    word_freq[w_norm] += 1

    # 2. Also check local data files if available
    for filename in ["pretrain.txt", "finetune.txt"]:
        filepath = os.path.join(DATA_CLEAN_DIR, filename)
        if os.path.exists(filepath):
            print(f"Reading {filepath}...")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            words = re.findall(r"[a-zA-ZàèéìíòóùúÀÈÉÌÍÒÓÙÚ']+", text)
            for w in words:
                w = normalize_word(w.strip("'"))
                if len(w) >= MIN_WORD_LENGTH:
                    word_freq[w] += 1

    return word_freq


SEED_WORDS = [
    # High-frequency poetic words
    "vita", "via", "gente", "mente", "parte", "morte", "luce", "voce",
    "mondo", "fondo", "secondo", "giocondo", "profondo", "tondo", "biondo",
    "terra", "guerra", "serra", "afferra", "atterra",
    "cielo", "velo", "gelo", "pelo", "stelo", "angelo", "modello",
    "amore", "dolore", "cuore", "errore", "fiore", "signore", "valore",
    "onore", "splendore", "ardore", "furore", "timore", "colore", "calore",
    "oscura", "paura", "natura", "misura", "figura", "ventura", "altura",
    "dura", "pura", "sicura", "matura", "futura", "avventura", "struttura",
    "smarrita", "ardita", "salita", "ferita", "unita", "gradita", "spedita",
    "infinita", "finita", "partita", "rapita", "scolpita", "ita", "udita", "punita",
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
    "padre", "madre", "leggiadre",
    "fuoco", "gioco", "loco", "poco",
    "alto", "salto", "assalto", "smalto",
    "ombra", "sgombra", "ingombra",
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
    "desio", "rio", "oblio", "natio", "mormorio", "addio",
    "via", "pria", "mia", "sia", "follia", "armonia", "poesia", "smarria", "maria",
    "trovai", "intrai", "abbandonai", "amai", "assai", "guardai", "omai",
    "trono", "dono", "suono", "buono", "tuono", "perdono",
    "ira", "mira", "gira", "sospira", "ammira", "aspira", "ispira",
    "volto", "molto", "sepolto", "ascolto", "sciolto", "raccolto",
    "forma", "norma", "riforma",
    "chiaro", "amaro", "raro", "caro", "avaro", "imparo",
    "ora", "ancora", "dimora", "aurora", "adora", "lavora", "divora",
    "speranza", "danza", "usanza", "possanza", "lontananza", "sembianza",
    "mente", "gente", "presente", "ardente", "corrente", "fremente",
    "potente", "vincente", "possente", "dolente", "silente", "prudente",
    "acceso", "peso", "inteso", "teso", "difeso", "offeso", "sospeso",
    "profeta", "poeta", "meta", "quieta", "lieta", "segreta",
    "eterno", "interno", "inferno", "inverno", "materno", "governo",
    "giorno", "ritorno", "intorno", "soggiorno", "contorno", "adorno",
    "atto", "fatto", "tratto", "ratto", "esatto", "contratto",
    "sogno", "bisogno", "vergogno",
    "pena", "serena", "piena", "catena", "scena", "arena",
    "speme", "teme", "geme", "insieme", "preme", "freme",
    "campo", "lampo", "scampo", "vampo",
    "punto", "giunto", "assunto", "defunto", "congiunto", "compunto",
    "spirto", "dritto", "scritto", "afflitto", "conflitto", "editto",
    "uomo", "duomo", "pomo",
    "causa", "pausa", "plausa",
    "grido", "nido", "lido", "fido",
    "onda", "sponda", "seconda", "gioconda", "profonda",
    "gloria", "storia", "memoria", "vittoria",
]


def build_rimario():
    """Build the comprehensive rimario and save to model/rimario.json."""
    print("=== Building Comprehensive Dante Rimario ===\n")

    # 1. Collect words from Commedia and corpus
    word_freq = extract_words_from_corpus()
    print(f"Total vocabulary collected: {len(word_freq):,} words\n")

    # 2. Add seed words (with initial frequency)
    for w in SEED_WORDS:
        w = normalize_word(w)
        if w not in word_freq:
            word_freq[w] = 5

    # 3. Group by stressed suffix
    suffix_groups = defaultdict(list)
    skipped = 0

    for word, freq in word_freq.items():
        suffix = get_stressed_suffix(word)
        if suffix is None:
            skipped += 1
            continue
        suffix_groups[suffix].append((word, freq))

    print(f"Suffixes identified: {len(suffix_groups):,}")
    print(f"Words skipped: {skipped:,}\n")

    # 4. Filter and sort
    rimario = {}
    total_words = 0

    for suffix in sorted(suffix_groups.keys()):
        entries = suffix_groups[suffix]
        if len(entries) < MIN_GROUP_SIZE:
            continue

        # Sort by frequency descending
        entries.sort(key=lambda x: -x[1])
        words = [w for w, _ in entries[:MAX_GROUP_SIZE]]

        # Deduplicate preserving order
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        if len(unique_words) >= MIN_GROUP_SIZE:
            rimario[suffix] = unique_words
            total_words += len(unique_words)

    print(f"Final Rhyme Families: {len(rimario):,}")
    print(f"Total Words in Rimario: {total_words:,}\n")

    # 5. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, "rimario.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rimario, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path)
    print(f"Saved rimario to {output_path} ({file_size / 1024:.1f} KB)")

    # 6. Sanity checks
    print("\n=== Sanity Checks ===")
    checks = [
        ("oscura", "paura", True),    # Inferno I, 2 vs 6
        ("dura", "paura", True),      # Inferno I, 4 vs 6
        ("vita", "smarrita", True),   # Inferno I, 1 vs 3
        ("forte", "morte", True),     # Inferno I, 5 vs 7
        ("trovai", "intrai", True),   # Inferno I, 8 vs 10
        ("punto", "giunto", True),    # Inferno I, 11 vs 13
        ("desio", "oblio", True),     # Dante -io rhyme
        ("via", "smarria", True),     # Dante -ia rhyme
        ("canto", "vento", False),    # False rhyme check
    ]
    
    all_ok = True
    for w1, w2, should_rhyme in checks:
        s1 = get_stressed_suffix(w1)
        s2 = get_stressed_suffix(w2)
        match = (s1 == s2)
        status = "[OK]" if match == should_rhyme else "[FAIL]"
        if match != should_rhyme:
            all_ok = False
        print(f"  {w1} (-{s1}) vs {w2} (-{s2}): match={match} (expected={should_rhyme}) {status}")

    if all_ok:
        print("\n[OK] All sanity checks passed!")
    else:
        print("\n[FAIL] Some checks failed!")

    return rimario


if __name__ == "__main__":
    build_rimario()
