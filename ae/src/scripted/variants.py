"""Opt-in finetuned variants of existing scripted strategies, for opponent
diversity in self-play.

CRITICAL: this module has NO import side-effects and NEVER mutates the global
STRATEGIES dict. Mutating STRATEGIES on import would silently change opponent
sampling, eval-pool composition, and any test that enumerates the dict for every
training run. Callers opt in via build_training_anchor_pool().

Variants reuse the base strategy's `layers` and `gates` and override only
`params` via dataclasses.replace (so the frozen _DEFAULT singleton is never
mutated). Only params verified to be consumed by the base strategy's layers are
tuned; suspected no-ops are excluded pending a behavioural audit.
"""
from dataclasses import replace

from scripted.strategies import STRATEGIES, Strategy


def _variant(base_name, suffix, **overrides):
    base = STRATEGIES[base_name]
    return Strategy(
        name=f"{base_name}_{suffix}",
        layers=base.layers,
        params=replace(base.params, **overrides),
        gates=base.gates,
    )


def _variants():
    return [
        _variant("adaptive", "aggressive", roi_gate_margin=0.08, vulture_hp_boost=2.6),
        _variant("adaptive", "conservative", roi_gate_margin=0.22, vulture_hp_boost=1.5),
        _variant("forager", "centric", centre_value_weight=-0.6, contested_value_factor=0.7),
        _variant("camper", "tight", camp_leash=3),
        _variant("camper", "loose", camp_leash=6),
        _variant("balanced_extreme_opening", "open_seeker",
                 openness_weight=2.5, breach_min_bombs=3),
        _variant("lean_rush", "trigger_happy", hunt_bomb_floor=2),
        _variant("lean_rush", "patient", hunt_bomb_floor=8),
        _variant("defender", "tight_perimeter", defend_radius=2),
        _variant("defender", "wide_perimeter", defend_radius=6),
    ]


def build_training_anchor_pool(include_variants: bool = True) -> dict[str, Strategy]:
    """Return a FRESH dict: all named STRATEGIES, plus variants when enabled.
    The global STRATEGIES dict is never mutated."""
    pool = dict(STRATEGIES)
    if include_variants:
        for v in _variants():
            if v.name in pool:
                raise ValueError(f"variant name clash: {v.name}")
            pool[v.name] = v
    return pool
