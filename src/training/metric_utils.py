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
VOWELS = set('aeiouàèéìíòóùúAEIOUÀÈÉÌÍÒÓÙÚ')
ACCENTED_VOWELS = set('àèéìíòóùúÀÈÉÌÍÒÓÙÚ')

# Common Italian diphthongs (count as 1 syllable)
DIPHTHONGS = [
    'ia', 'ie', 'io', 'iu',
    'ua', 'ue', 'ui', 'uo',
    'ai', 'ei', 'oi', 'ui', 'au', 'eu',
]

# Hiatus patterns (count as 2 syllables)
HIATUS_PATTERNS = [
    'ìa', 'ìe', 'ìo', 'ìu',
    'aì', 'eì', 'oì', 'uì',
    'ùa', 'ùe', 'ùo', 'ùi',
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
        word = re.sub(r"[^\w'àèéìíòóùú]", '', word)
        if not word:
            return 0
        
        syllables = 0
        i = 0
        
        while i < len(word):
            if word[i] in VOWELS or word[i] in 'àèéìíòóùú':
                syllables += 1
                
                # Check for diphthong (doesn't add extra syllable)
                if i + 1 < len(word):
                    pair = word[i:i+2].lower()
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
                next_starts_vowel = words[i+1] and words[i+1][0] in VOWELS
                
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
            return word[vowel_positions[-2]:]
        elif vowel_positions:
            return word[vowel_positions[-1]:]
        else:
            return word[-3:] if len(word) >= 3 else word
    
    def rhymes_with(self, verse1: str, verse2: str, strict: bool = False) -> bool:
        """
        Check if two verses rhyme.
        
        Args:
            verse1: First verse
            verse2: Second verse
            strict: If True, require exact suffix match
        """
        suffix1 = self.get_rhyme_suffix(verse1)
        suffix2 = self.get_rhyme_suffix(verse2)
        
        if not suffix1 or not suffix2:
            return False
        
        if strict:
            return suffix1 == suffix2
        
        # Lenient matching: check if endings are similar
        min_len = min(len(suffix1), len(suffix2), self.min_suffix_len)
        return suffix1[-min_len:] == suffix2[-min_len:]
    
    def get_rhyme_scheme(self, verses: List[str]) -> List[str]:
        """
        Determine the rhyme scheme of a list of verses.
        Returns a list like ['A', 'B', 'A', 'B', 'C', 'B', ...]
        """
        if not verses:
            return []
        
        scheme = []
        suffix_to_letter = {}
        current_letter = 'A'
        
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
    
    def __init__(self, 
                 syllable_weight: float = 0.15,  # Reduced: user cares less about syllables
                 rhyme_weight: float = 0.55,     # Increased: ABA BCB CDC is key
                 structure_weight: float = 0.30, # Increased: proper tercet structure
                 syllable_tolerance: int = 2):
        self.syllable_counter = ItalianSyllableCounter()
        self.rhyme_detector = RhymeDetector()
        
        self.syllable_weight = syllable_weight
        self.rhyme_weight = rhyme_weight
        self.structure_weight = structure_weight
        self.syllable_tolerance = syllable_tolerance
    
    def split_into_verses(self, text: str) -> List[str]:
        """Split text into verses (lines)."""
        lines = text.strip().split('\n')
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
        """
        if len(verses) < 3:
            return 0.0
        
        # Expected pattern for terzina incatenata
        # ABA BCB CDC DED ...
        # Position 0,2 rhyme (A), position 1,3,5 rhyme (B), etc.
        
        scheme = self.rhyme_detector.get_rhyme_scheme(verses)
        
        correct = 0
        total = 0
        
        # Check terzina pattern
        for i in range(0, len(verses) - 2, 3):
            if i + 2 < len(scheme):
                # Check ABA pattern within tercet
                if scheme[i] == scheme[i + 2]:
                    correct += 1
                total += 1
                
                # Check chain: B of this tercet should match outer of next
                if i + 3 < len(scheme) and i + 4 < len(scheme):
                    if scheme[i + 1] == scheme[i + 3]:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
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
            return 0.0, {'syllables': 0.0, 'rhyme': 0.0, 'structure': 0.0, 'verses': len(verses)}
        
        syllable_score = self.score_syllables(verses)
        rhyme_score = self.score_rhyme_scheme(verses)
        structure_score = self.score_structure(verses)
        
        total = (
            self.syllable_weight * syllable_score +
            self.rhyme_weight * rhyme_score +
            self.structure_weight * structure_score
        )
        
        breakdown = {
            'syllables': syllable_score,
            'rhyme': rhyme_score,
            'structure': structure_score,
            'total': total,
            'verses': len(verses)
        }
        
        return total, breakdown


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


if __name__ == '__main__':
    # Test examples
    print("=== Italian Metric Utilities Test ===\n")
    
    # Test syllable counting
    sc = ItalianSyllableCounter()
    test_verses = [
        "Nel mezzo del cammin di nostra vita",  # 11
        "mi ritrovai per una selva oscura",      # 11
        "ché la diritta via era smarrita",       # 11
        "Ahi quanto a dir qual era è cosa dura", # 11
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
