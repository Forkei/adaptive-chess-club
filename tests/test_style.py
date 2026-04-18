from __future__ import annotations

from app.characters.style import StyleFragments, style_summary_line, style_to_prompt_fragments
from app.models.character import Character


def _char(**overrides) -> Character:
    defaults: dict = dict(
        name="Test",
        aggression=5,
        risk_tolerance=5,
        patience=5,
        trash_talk=5,
        target_elo=1500,
    )
    defaults.update(overrides)
    return Character(**defaults)


def test_fragments_shape_is_stable():
    frags = style_to_prompt_fragments(_char())
    assert set(frags.keys()) == set(StyleFragments.__annotations__.keys())
    for v in frags.values():
        assert isinstance(v, str) and v


def test_low_mid_high_buckets_produce_distinct_aggression_fragments():
    low = style_to_prompt_fragments(_char(aggression=1))["aggression"]
    mid = style_to_prompt_fragments(_char(aggression=5))["aggression"]
    high = style_to_prompt_fragments(_char(aggression=10))["aggression"]
    assert low != mid != high
    assert low != high


def test_bucket_boundaries():
    # 3 is low, 4 is mid; 7 is mid, 8 is high. This encodes the mapping so a
    # future edit doesn't silently shift the buckets.
    assert (
        style_to_prompt_fragments(_char(aggression=3))["aggression"]
        == style_to_prompt_fragments(_char(aggression=1))["aggression"]
    )
    assert (
        style_to_prompt_fragments(_char(aggression=4))["aggression"]
        == style_to_prompt_fragments(_char(aggression=7))["aggression"]
    )
    assert (
        style_to_prompt_fragments(_char(aggression=8))["aggression"]
        == style_to_prompt_fragments(_char(aggression=10))["aggression"]
    )


def test_summary_line_uses_character_name():
    line = style_summary_line(_char(name="Vera"))
    assert line.startswith("Vera ")
    assert line.endswith(".")


def test_each_slider_changes_only_its_own_fragment():
    base = style_to_prompt_fragments(_char())
    with_high_patience = style_to_prompt_fragments(_char(patience=10))

    assert with_high_patience["patience"] != base["patience"]
    assert with_high_patience["aggression"] == base["aggression"]
    assert with_high_patience["risk_tolerance"] == base["risk_tolerance"]
    assert with_high_patience["trash_talk"] == base["trash_talk"]
