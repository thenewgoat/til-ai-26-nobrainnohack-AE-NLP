from scripted.strategies import STRATEGIES
from scripted.variants import build_training_anchor_pool, _variants


def test_pool_includes_all_base_strategies():
    pool = build_training_anchor_pool(include_variants=False)
    for name in STRATEGIES:
        assert name in pool


def test_pool_includes_variants_when_enabled():
    pool = build_training_anchor_pool(include_variants=True)
    assert "adaptive_aggressive" in pool
    assert "camper_tight" in pool
    assert len(pool) == len(STRATEGIES) + len(_variants())


def test_pool_excludes_variants_when_disabled():
    pool = build_training_anchor_pool(include_variants=False)
    assert "adaptive_aggressive" not in pool
    assert len(pool) == len(STRATEGIES)


def test_global_strategies_dict_is_never_mutated():
    before = dict(STRATEGIES)
    build_training_anchor_pool(include_variants=True)
    assert STRATEGIES == before
    assert "adaptive_aggressive" not in STRATEGIES


def test_each_variant_differs_from_its_base_params():
    for v in _variants():
        parts = v.name.split("_")
        base = None
        for k in range(len(parts) - 1, 0, -1):
            cand = "_".join(parts[:k])
            if cand in STRATEGIES:
                base = STRATEGIES[cand]
                break
        assert base is not None, v.name
        assert v.params != base.params          # override actually changed something
        assert v.params is not base.params      # replace() produced a distinct instance
        assert v.layers is base.layers          # layers reused, not rebuilt
        assert v.gates is base.gates


def test_base_strategy_params_objects_unchanged_after_pool_build():
    before = {name: STRATEGIES[name].params for name in STRATEGIES}
    build_training_anchor_pool(include_variants=True)
    for name in STRATEGIES:
        # same object identity → building the pool never mutated/replaced base params
        assert STRATEGIES[name].params is before[name]
