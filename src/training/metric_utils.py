"""
Italian Metric Utilities
========================
Tools for evaluating Dante's terzina incatenata metric:
- Syllable counting with sinalefe rules
- Rhyme detection for ABA BCB CDC scheme
- Terzina scoring for RL reward computation
"""

import re
from typing import List, Tuple, Optional


# Italian vowels (including accented)
VOWELS = set("aeiouàèéìíòóùúAEIOUÀÈÉÌÍÒÓÙÚ")
ACCENTED_VOWELS = set("àèéìíòóùúÀÈÉÌÍÒÓÙÚ")

# Common Italian diphthongs (count as 1 syllable)
DIPHTHONGS = [
    "ia",
    "ie",
    "io",
    "iu",
    "ua",
    "ue",
    "ui",
    "uo",
    "ai",
    "ei",
    "oi",
    "ui",
    "au",
    "eu",
]

# Hiatus patterns (count as 2 syllables)
HIATUS_PATTERNS = [
    "ìa",
    "ìe",
    "ìo",
    "ìu",
    "aì",
    "eì",
    "oì",
    "uì",
    "ùa",
    "ùe",
    "ùo",
    "ùi",
]


class ItalianSyllableCounter:
    """
    Count syllables in Italian text with poetic rules.

    Handles:
    - Basic vowel counting
    - Diphthongs and hiatus
    - Sinalefe (elision between words)
    - Apostrophe elision
    """

    def __init__(self, apply_sinalefe: bool = True):
        self.apply_sinalefe = apply_sinalefe

    def count_word_syllables(self, word: str) -> int:
        """Count syllables in a single word."""
        word = word.lower().strip()
        if not word:
            return 0

        # Remove punctuation but keep apostrophes for now
        word = re.sub(r"[^\w'àèéìíòóùú]", "", word)
        if not word:
            return 0

        syllables = 0
        i = 0

        while i < len(word):
            if word[i] in VOWELS or word[i] in "àèéìíòóùú":
                syllables += 1

                # Check for diphthong (doesn't add extra syllable)
                if i + 1 < len(word):
                    pair = word[i : i + 2].lower()
                    # Check if it's a hiatus (add syllable) or diphthong (don't)
                    is_hiatus = any(h in pair for h in HIATUS_PATTERNS)
                    is_diphthong = pair in DIPHTHONGS

                    if is_diphthong and not is_hiatus:
                        i += 1  # Skip next vowel, it's part of diphthong

            i += 1

        return max(syllables, 1) if word else 0

    def count_verse_syllables(self, verse: str) -> int:
        """
        Count syllables in a verse (line of poetry).
        Applies sinalefe between words if enabled.
        """
        # Clean the verse
        verse = verse.strip()
        if not verse:
            return 0

        # Split into words, keeping track of positions
        words = re.findall(r"[\w'àèéìíòóùú]+", verse.lower())
        if not words:
            return 0

        total = 0

        for i, word in enumerate(words):
            word_syllables = self.count_word_syllables(word)
            total += word_syllables

            # Apply sinalefe: if word ends with vowel and next starts with vowel
            if self.apply_sinalefe and i < len(words) - 1:
                current_ends_vowel = word and word[-1] in VOWELS
                next_starts_vowel = words[i + 1] and words[i + 1][0] in VOWELS

                # Also check for elision with apostrophe
                current_ends_apostrophe = "'" in word

                if (current_ends_vowel or current_ends_apostrophe) and next_starts_vowel:
                    total -= 1  # Sinalefe: merge the vowels

        return max(total, 1)

    def is_endecasillabo(self, verse: str, tolerance: int = 1) -> Tuple[bool, int]:
        """
        Check if a verse is an endecasillabo (11 syllables).

        Returns:
            Tuple of (is_valid, syllable_count)
        """
        count = self.count_verse_syllables(verse)
        is_valid = abs(count - 11) <= tolerance
        return is_valid, count


class RhymeDetector:
    """
    Detect rhymes in Italian poetry.

    Extracts the rhyming suffix (from last stressed vowel to end)
    and compares verses for rhyme matching.
    """

    def __init__(self, min_suffix_len: int = 2):
        self.min_suffix_len = min_suffix_len

    def get_rhyme_suffix(self, verse: str) -> str:
        """
        Extract the rhyming suffix of a verse.
        This is approximately from the last stressed vowel to the end.
        """
        # Get last word
        words = re.findall(r"[\w'àèéìíòóùú]+", verse.lower())
        if not words:
            return ""

        last_word = words[-1]

        # Find last accented vowel, or guess stress position
        # In Italian, stress is usually on penultimate syllable
        suffix = self._extract_suffix(last_word)

        return suffix

    def _extract_suffix(self, word: str) -> str:
        """Extract rhyming suffix from a word."""
        word = word.lower().strip()
        if len(word) < 2:
            return word

        # Look for explicit accent
        for i, char in enumerate(word):
            if char in ACCENTED_VOWELS:
                return word[i:]

        # No explicit accent: find last vowel cluster
        # Italian words usually stress penultimate syllable
        vowel_positions = [i for i, c in enumerate(word) if c in VOWELS]

        if len(vowel_positions) >= 2:
            # Take from penultimate vowel
            return word[vowel_positions[-2] :]
        elif vowel_positions:
            return word[vowel_positions[-1] :]
        else:
            return word[-3:] if len(word) >= 3 else word

    def _get_vowels(self, text: str) -> str:
        """Extract only vowels from text."""
        return "".join([c for c in text if c in VOWELS or c in ACCENTED_VOWELS])

    def score_rhyme(self, verse1: str, verse2: str) -> float:
        """
        Score how well two verses rhyme.
        Returns:
            1.0 if strict rhyme (suffix match)
            0.6 if assonance (vowels match)
            0.0 otherwise
        """
        suffix1 = self.get_rhyme_suffix(verse1)
        suffix2 = self.get_rhyme_suffix(verse2)

        if not suffix1 or not suffix2:
            return 0.0

        min_len = min(len(suffix1), len(suffix2), self.min_suffix_len)
        s1_end = suffix1[-min_len:]
        s2_end = suffix2[-min_len:]

        # Strict rhyme
        if s1_end == s2_end:
            return 1.0

        # Assonance check (vowels only)
        v1 = self._get_vowels(suffix1)
        v2 = self._get_vowels(suffix2)

        # Compare last few vowels
        min_vowels = min(len(v1), len(v2), 2)
        if min_vowels > 0 and v1[-min_vowels:] == v2[-min_vowels:]:
            return 0.6  # Partial credit for assonance

        return 0.0

    def rhymes_with(self, verse1: str, verse2: str, strict: bool = False) -> bool:
        """
        Check if two verses rhyme.

        Args:
            verse1: First verse
            verse2: Second verse
            strict: If True, require exact suffix match
        """
        return self.score_rhyme(verse1, verse2) >= (1.0 if strict else 0.6)

    def get_rhyme_scheme(self, verses: List[str]) -> List[str]:
        """
        Determine the rhyme scheme of a list of verses.
        Returns a list like ['A', 'B', 'A', 'B', 'C', 'B', ...]
        """
        if not verses:
            return []

        scheme = []
        suffix_to_letter = {}
        current_letter = "A"

        for verse in verses:
            suffix = self.get_rhyme_suffix(verse)

            # Check if this suffix matches any existing
            matched = False
            for existing_suffix, letter in suffix_to_letter.items():
                if self._suffixes_rhyme(suffix, existing_suffix):
                    scheme.append(letter)
                    matched = True
                    break

            if not matched:
                suffix_to_letter[suffix] = current_letter
                scheme.append(current_letter)
                current_letter = chr(ord(current_letter) + 1)

        return scheme

    def _suffixes_rhyme(self, s1: str, s2: str) -> bool:
        """Check if two suffixes rhyme."""
        if not s1 or not s2:
            return False
        min_len = min(len(s1), len(s2), self.min_suffix_len)
        return s1[-min_len:] == s2[-min_len:]


class TerzinaScorer:
    """
    Score generated text for adherence to terzina dantesca rules.

    Terzina incatenata (ABA BCB CDC...):
    - Groups of 3 verses (tercets)
    - Each verse is an endecasillabo (11 syllables)
    - Rhyme scheme links tercets: middle verse rhymes with outer verses of next tercet
    """

    def __init__(
        self,
        syllable_weight: float = 0.20,  # Reduced: user cares less about syllables
        rhyme_weight: float = 0.4,   #Increased: ABA BCB CDC is key
        structure_weight: float = 0.30,  # Increased: proper tercet structure
        repetition_penalty_weight: float = 0.9,  # NEW: penalty for repetitive text
        syllable_tolerance: int = 2,
    ):
        self.syllable_counter = ItalianSyllableCounter()
        self.rhyme_detector = RhymeDetector()

        self.syllable_weight = syllable_weight
        self.rhyme_weight = rhyme_weight
        self.structure_weight = structure_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.syllable_tolerance = syllable_tolerance

    def compute_repetition_penalty(self, text: str) -> float:
        """
        Compute a penalty for repetitive text (prevents reward hacking).
        
        Returns a penalty value between 0 (no repetition) and 1 (heavy repetition).
        Now much more aggressive to prevent "mosso mosso mosso" type outputs.
        """
        from collections import Counter
        
        words = text.lower().split()
        if len(words) < 4:
            return 0.0
        
        penalties = []
        
        # === 1. Single word repetition check (new, most aggressive) ===
        # If any single word appears too many times, heavy penalty
        word_counts = Counter(words)
        total_words = len(words)
        
        # Filter out common short words that may repeat naturally
        stop_words = {'e', 'il', 'la', 'lo', 'i', 'le', 'gli', 'un', 'una', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'che', 'non', 'mi', 'si', 'è'}
        
        for word, count in word_counts.items():
            if word in stop_words or len(word) < 3:
                continue
            
            word_ratio = count / total_words
            # If a single word is >15% of total words, that's suspicious
            # If it's >25%, that's very bad
            if word_ratio > 0.15:
                # Exponential penalty for word over-use
                word_penalty = min(1.0, (word_ratio - 0.10) * 5)  # Scales 0.15->0.25, 0.25->0.75, 0.30->1.0
                penalties.append(word_penalty)
        
        # === 2. N-gram repetition check (improved) ===
        for n in [2, 3, 4]:
            if len(words) < n * 2:
                continue
            
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                continue
                
            counts = Counter(ngrams)
            
            # Calculate repetition ratio with stronger weighting
            total = len(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            ratio = repeated / total if total > 0 else 0
            
            # Much stronger weight for repeated phrases
            # n=2: ratio * 1.5, n=3: ratio * 2.5, n=4: ratio * 4
            weight = n * 1.0 + (n - 2) * 0.5
            penalties.append(ratio * weight)
        
        # === 3. Consecutive duplicate check (new) ===
        # Check for words repeated directly in sequence: "mosso mosso mosso"
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i+1] and words[i] not in stop_words and len(words[i]) >= 3:
                consecutive_repeats += 1
        
        if consecutive_repeats > 0:
            consec_ratio = consecutive_repeats / len(words)
            # Very aggressive: even 2 consecutive repeats in short text is bad
            consec_penalty = min(1.0, consec_ratio * 10)
            penalties.append(consec_penalty)
        
        if not penalties:
            return 0.0
        
        # Take max penalty (worst repetition type) but also add average for combined effect
        max_penalty = max(penalties)
        avg_penalty = sum(penalties) / len(penalties)
        
        # Final penalty: 70% worst offender + 30% average
        final_penalty = 0.7 * max_penalty + 0.3 * avg_penalty
        
        return min(1.0, final_penalty)

    def split_into_verses(self, text: str) -> List[str]:
        """Split text into verses (lines)."""
        lines = text.strip().split("\n")
        # Filter empty lines and strip whitespace
        verses = [line.strip() for line in lines if line.strip()]
        return verses

    def score_syllables(self, verses: List[str]) -> float:
        """
        Score verses for syllable count (target: 11).
        Returns a score from 0 to 1.
        """
        if not verses:
            return 0.0

        scores = []
        for verse in verses:
            count = self.syllable_counter.count_verse_syllables(verse)
            # Score based on distance from 11
            deviation = abs(count - 11)
            if deviation <= self.syllable_tolerance:
                score = 1.0 - (deviation / (self.syllable_tolerance + 1))
            else:
                score = max(0, 0.5 - (deviation - self.syllable_tolerance) * 0.1)
            scores.append(score)

        return sum(scores) / len(scores)

    def score_rhyme_scheme(self, verses: List[str]) -> float:
        """
        Score adherence to ABA BCB CDC rhyme scheme.
        Returns a score from 0 to 1.
        Using partial credit for assonance to guide learning.
        """
        if len(verses) < 3:
            return 0.0

        # Expected pattern for terzina incatenata
        # ABA BCB CDC DED ...
        # Position 0,2 rhyme (A), position 1,3,5 rhyme (B), etc.

        total_score = 0
        total_checks = 0

        for i in range(0, len(verses) - 2, 3):
            # Check ABA pattern within tercet: verse i and i+2
            if i + 2 < len(verses):
                score = self.rhyme_detector.score_rhyme(verses[i], verses[i + 2])
                total_score += score
                total_checks += 1

            # Check chain: verses[i+1] (B) matches verses[i+3] (B)
            # Link B of this tercet to outer of next
            if i + 3 < len(verses):
                score = self.rhyme_detector.score_rhyme(verses[i + 1], verses[i + 3])
                total_score += score
                total_checks += 1

        return total_score / total_checks if total_checks > 0 else 0.0

    def score_structure(self, verses: List[str]) -> float:
        """
        Score structural elements (proper verse separation, etc.)
        Returns a score from 0 to 1.
        """
        if not verses:
            return 0.0

        scores = []

        # Reward having groups of 3 verses
        num_complete_tercets = len(verses) // 3
        if num_complete_tercets > 0:
            scores.append(min(1.0, num_complete_tercets / 3))  # Reward up to 3 tercets

        # Reward reasonable verse lengths (not too short, not too long)
        for verse in verses:
            word_count = len(verse.split())
            if 5 <= word_count <= 15:
                scores.append(1.0)
            elif 3 <= word_count < 5 or 15 < word_count <= 20:
                scores.append(0.5)
            else:
                scores.append(0.2)

        return sum(scores) / len(scores) if scores else 0.0

    def compute_reward(self, text: str) -> Tuple[float, dict]:
        """
        Compute the total reward for a generated text.

        Returns:
            Tuple of (total_reward, breakdown_dict)
        """
        verses = self.split_into_verses(text)

        if len(verses) < 3:
            # Give small reward instead of 0 to provide learning gradient
            # This encourages model to generate more verses rather than collapsing
            small_reward = 0.02 * len(verses)  # 0.02 for 1 verse, 0.04 for 2 verses
            return small_reward, {"syllables": 0.0, "rhyme": 0.0, "structure": 0.0, "repetition": 0.0, "verses": len(verses)}

        syllable_score = self.score_syllables(verses)
        rhyme_score = self.score_rhyme_scheme(verses)
        structure_score = self.score_structure(verses)
        
        # Compute repetition penalty
        repetition_penalty = self.compute_repetition_penalty(text)

        total = (
            self.syllable_weight * syllable_score
            + self.rhyme_weight * rhyme_score
            + self.structure_weight * structure_score
        )
        
        # Apply repetition penalty (cap at 70% reduction to maintain learning signal)
        if repetition_penalty > 0.1:
            # Cap the penalty effect at 70% reduction (was 90%)
            penalty_multiplier = max(0.3, 1 - 0.7 * repetition_penalty)
            total = total * penalty_multiplier
        
        # Minimum reward floor to prevent complete collapse
        # Even heavily penalized text gets some reward to guide learning
        total = max(0.05, total)

        breakdown = {
            "syllables": syllable_score,
            "rhyme": rhyme_score,
            "structure": structure_score,
            "repetition": repetition_penalty,
            "total": total,
            "verses": len(verses),
        }

        return total, breakdown

    def compute_per_verse_rewards(self, text: str) -> List[Tuple[float, dict]]:
        """
        Compute reward for each verse individually (for reward shaping).
        
        Returns:
            List of (reward, breakdown) tuples for each verse.
            This enables denser learning signal during RL training.
        """
        verses = self.split_into_verses(text)
        verse_rewards = []
        
        for i, verse in enumerate(verses):
            breakdown = {}
            
            # Syllable score for this verse
            count = self.syllable_counter.count_verse_syllables(verse)
            deviation = abs(count - 11)
            if deviation <= self.syllable_tolerance:
                syl_score = 1.0 - (deviation / (self.syllable_tolerance + 1))
            else:
                syl_score = max(0, 0.5 - (deviation - self.syllable_tolerance) * 0.1)
            breakdown["syllables"] = syl_score
            breakdown["syllable_count"] = count
            
            # Structure score for this verse
            word_count = len(verse.split())
            if 5 <= word_count <= 15:
                struct_score = 1.0
            elif 3 <= word_count < 5 or 15 < word_count <= 20:
                struct_score = 0.5
            else:
                struct_score = 0.2
            breakdown["structure"] = struct_score
            
            # Rhyme score (check against previous verses in tercet pattern)
            rhyme_score = 0.0
            tercet_pos = i % 3  # Position within tercet: 0, 1, 2
            
            # ABA pattern: verse 0 should rhyme with verse 2
            if tercet_pos == 2 and i >= 2:
                # Check rhyme with verse i-2 (first of tercet)
                rhyme_score = self.rhyme_detector.score_rhyme(verses[i-2], verse)
            # Chain: middle verse (pos 1) should rhyme with first verse of next tercet
            # We can't check forward, but we can check backward
            elif tercet_pos == 0 and i >= 2:
                # First of new tercet should rhyme with middle of previous (i-2)
                rhyme_score = self.rhyme_detector.score_rhyme(verses[i-2], verse)
            
            breakdown["rhyme"] = rhyme_score
            
            # Compute weighted verse reward
            verse_reward = (
                self.syllable_weight * syl_score +
                self.rhyme_weight * rhyme_score +
                self.structure_weight * struct_score
            )
            breakdown["total"] = verse_reward
            
            verse_rewards.append((verse_reward, breakdown))
        
        return verse_rewards


# Convenience functions for quick testing
def count_syllables(text: str) -> int:
    """Quick syllable count for a verse."""
    return ItalianSyllableCounter().count_verse_syllables(text)


def check_rhyme(verse1: str, verse2: str) -> bool:
    """Quick rhyme check between two verses."""
    return RhymeDetector().rhymes_with(verse1, verse2)


def score_terzina(text: str) -> float:
    """Quick score for a terzina text."""
    return TerzinaScorer().compute_reward(text)[0]


if __name__ == "__main__":
    # Test examples
    print("=== Italian Metric Utilities Test ===\n")

    # Test syllable counting
    sc = ItalianSyllableCounter()
    test_verses = [
        "Nel mezzo del cammin di nostra vita",  # 11
        "mi ritrovai per una selva oscura",  # 11
        "ché la diritta via era smarrita",  # 11
        "Ahi quanto a dir qual era è cosa dura",  # 11
    ]

    print("Syllable Counting:")
    for verse in test_verses:
        count = sc.count_verse_syllables(verse)
        is_valid, _ = sc.is_endecasillabo(verse)
        print(f"  '{verse[:40]}...' -> {count} syllables {'✓' if is_valid else '✗'}")

    # Test rhyme detection
    rd = RhymeDetector()
    print("\nRhyme Detection:")
    print(f"  'vita' suffix: {rd.get_rhyme_suffix('vita')}")
    print(f"  'smarrita' suffix: {rd.get_rhyme_suffix('smarrita')}")
    print(f"  'vita' rhymes with 'smarrita': {rd.rhymes_with('vita', 'smarrita')}")

    # Test terzina scoring
    ts = TerzinaScorer()
    sample_text = """Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura
ché la diritta via era smarrita
Ahi quanto a dir qual era è cosa dura
esta selva selvaggia e aspra e forte
che nel pensier rinova la paura
Tant'è amara che poco è più morte
ma per trattar del ben ch'i' vi trovai
dirò de l'altre cose ch'i' v'ho scorte"""

    print("\nTerzina Scoring (Inferno I, 1-9):")
    reward, breakdown = ts.compute_reward(sample_text)
    print(f"  Total reward: {reward:.3f}")
    for key, value in breakdown.items():
        print(f"    {key}: {value:.3f}" if isinstance(value, float) else f"    {key}: {value}")
