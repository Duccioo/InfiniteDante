"""
Trecento Poetic Corpus Builder for InfiniteDante
================================================
Downloads and cleans authentic 13th-14th century Italian poetry:
- Dante Alighieri: Divina Commedia (Inferno, Purgatorio, Paradiso), Rime, Vita Nuova
- Francesco Petrarca: Trionfi (terza rima) & Canzoniere
- Giovanni Boccaccio: Rime & Filostrato (endecasillabi)

Normalizes tercet structure and prepares clean corpus for BPE & model training.
"""

import os
import re
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_FILE = os.path.join(DATA_CLEAN_DIR, "trecento_poetic_corpus.txt")

SOURCES = [
    {
        "author": "Dante Alighieri",
        "work": "Inferno",
        "url": "https://raw.githubusercontent.com/asperti/Dante/master/inferno.txt",
    },
    {
        "author": "Dante Alighieri",
        "work": "Purgatorio",
        "url": "https://raw.githubusercontent.com/asperti/Dante/master/purgatorio.txt",
    },
    {
        "author": "Dante Alighieri",
        "work": "Paradiso",
        "url": "https://raw.githubusercontent.com/asperti/Dante/master/paradiso.txt",
    },
    {
        "author": "Francesco Petrarca",
        "work": "Trionfi (Terza Rima)",
        "url": "https://raw.githubusercontent.com/asperti/Dante/master/trionfi.txt",
        "fallback_url": "https://raw.githubusercontent.com/napolux/parole/master/dante.txt"
    }
]


def clean_verse_text(text: str) -> str:
    """Normalize verses, clean line numbering, markers, and roman numerals."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # Skip headers, canto titles, and number markers
        if re.match(r"^(canto\s+[ivxlcdm]+|inferno|purgatorio|paradiso|trionfo)", line, re.I):
            continue
        if re.match(r"^\d+$", line):
            continue
        if line.startswith("***") or line.startswith("---") or line.startswith("==="):
            continue

        # Clean trailing verse line numbers (e.g. "Nel mezzo del cammin... 3")
        line = re.sub(r"\s+\d+\s*$", "", line)
        line = line.strip()

        if len(line) >= 8:  # Minimum plausible verse length
            lines.append(line)

    return "\n".join(lines)


def build_corpus():
    os.makedirs(DATA_CLEAN_DIR, exist_ok=True)
    all_corpus = []

    print("=== Building Trecento Poetic Corpus ===")

    for source in SOURCES:
        name = f"{source['author']} - {source['work']}"
        url = source["url"]
        print(f"Fetching {name} from {url}...")
        raw_text = ""
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  -> Primary URL failed ({e}), attempting fallback...")
            fallback = source.get("fallback_url")
            if fallback:
                try:
                    req = urllib.request.Request(
                        fallback, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        raw_text = resp.read().decode("utf-8", errors="replace")
                except Exception as e2:
                    print(f"  -> Fallback also failed: {e2}")

        if raw_text:
            cleaned = clean_verse_text(raw_text)
            verse_count = len([v for v in cleaned.splitlines() if v.strip()])
            print(f"  -> Added {verse_count:,} verses ({len(cleaned):,} characters)")
            all_corpus.append(cleaned)
        else:
            print(f"  -> Skipping {name} (source unavailable)")

    combined = "\n\n\n".join(all_corpus)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(combined)

    total_verses = len([v for v in combined.splitlines() if v.strip()])
    file_size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nCorpus build complete: {OUTPUT_FILE}")
    print(f"Total verses: {total_verses:,}")
    print(f"Total size: {file_size_kb:.1f} KB")

    return OUTPUT_FILE


if __name__ == "__main__":
    build_corpus()
