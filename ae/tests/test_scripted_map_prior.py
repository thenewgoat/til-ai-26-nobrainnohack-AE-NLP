from scripted.map_prior import MapPrior


def test_loads_default_map():
    m = MapPrior.load()
    assert m.grid_size == 16
    assert len(m.collectibles) > 0          # dict (x,y) -> total value
    assert len(m.wall_between) == 197       # frozenset pair -> destructible bool


def test_identify_team_exact_match():
    m = MapPrior.load()
    # team 0 base is (13, 9), team 3 base is (2, 6).
    assert m.identify_team((13, 9)) == 0
    assert m.identify_team((2, 6)) == 3


def test_identify_team_fallback_to_nearest():
    m = MapPrior.load()
    # (12, 9) is closest to team 0's base (13, 9).
    assert m.identify_team((12, 9)) == 0


def test_team_relative_accessors():
    m = MapPrior.load()
    m.identify_team((13, 9))                # we are team 0
    assert m.our_base == (13, 9)
    assert (13, 9) not in m.enemy_bases
    assert len(m.enemy_bases) == 5


def test_collectible_values():
    m = MapPrior.load()
    # mission=5, recon=1, resource=2; a cell may stack kinds.
    assert all(v > 0 for v in m.collectibles.values())
