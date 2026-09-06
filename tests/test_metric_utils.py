"""
Unit tests for Dante metric utilities and phonetics
"""

import os
import sys
import pytest

# Add src/training to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "training"))
from metric_utils import (
    count_syllables,
    get_stressed_suffix,
    check_rhyme,
    RhymeDetector,
    TerzinaScorer,
)


def test_dante_famous_verses_syllables():
    v1 = "Nel mezzo del cammin di nostra vita"
    assert count_syllables(v1) == 11

    v2 = "mi ritrovai per una selva oscura"
    assert count_syllables(v2) == 11

    v3 = "ché la diritta via era smarrita"
    assert count_syllables(v3) == 11

    v4 = "Ahi quanto a dir qual era è cosa dura"
    assert count_syllables(v4) == 11

    v5 = "che nel pensier rinova la paura"
    assert count_syllables(v5) == 11


def test_dante_rhyme_hiatus():
    # 'paura', 'dura', 'oscura' must all rhyme on 'ura'
    r1 = get_stressed_suffix("oscura")
    r2 = get_stressed_suffix("dura")
    r3 = get_stressed_suffix("paura")

    assert r1 == "ura"
    assert r2 == "ura"
    assert r3 == "ura"
    assert check_rhyme("oscura", "paura") is True
    assert check_rhyme("dura", "paura") is True


def test_dante_rhyme_oxytone():
    # 'trovai' and 'intrai' must rhyme on 'ai'
    r1 = get_stressed_suffix("trovai")
    r2 = get_stressed_suffix("intrai")

    assert r1 == "ai"
    assert r2 == "ai"
    assert check_rhyme("trovai", "intrai") is True


def test_terzina_scorer():
    scorer = TerzinaScorer()

    canto1_t1 = (
        "Nel mezzo del cammin di nostra vita\n"
        "mi ritrovai per una selva oscura\n"
        "ché la diritta via era smarrita"
    )
    score, breakdown = scorer.compute_reward(canto1_t1)
    assert score > 0.7
    assert breakdown["rhyme"] > 0.8
