"""Static novice-map prior loaded from arena_map.json, plus team identification."""
import json
from pathlib import Path

from scripted.geometry import MOVE

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "arena_map.json"
_KIND_VALUE = {"mission": 5.0, "recon": 1.0, "resource": 2.0}


def _edge_pair(ax, ay, direction):
    """Frozenset of the two tiles a wall edge separates."""
    dx, dy = MOVE[direction]
    return frozenset({(ax, ay), (ax + dx, ay + dy)})


class MapPrior:
    """Immutable static map. Call identify_team() once per episode at step 0."""

    def __init__(self, grid_size, wall_between, bases, spawns, collectibles):
        self.grid_size = grid_size
        self.wall_between = wall_between        # frozenset{tileA,tileB} -> destructible bool
        self.bases = bases                      # team int -> (x, y)
        self.spawns = spawns                    # team int -> {"pos": (x,y), "facing": int}
        self.collectibles = collectibles        # (x, y) -> total value float
        self.team = None
        self.our_base = None
        self.enemy_bases = []
        # Resource-kind collectible cells (populated by MapPrior.load).
        # `None` means "kind info unavailable" — consumers (e.g. sweep) fall
        # back to all collectibles in that case, keeping hand-rolled priors
        # in tests working.
        self.resource_cells = None

    @classmethod
    def load(cls, path=_DEFAULT_PATH):
        data = json.loads(Path(path).read_text())
        wall_between = {}
        for ax, ay, direction, destructible in data["walls"]:
            wall_between[_edge_pair(ax, ay, direction)] = bool(destructible)
        bases = {int(t): tuple(p) for t, p in data["bases"].items()}
        spawns = {int(t): {"pos": tuple(s["pos"]), "facing": int(s["facing"])}
                  for t, s in data["spawns"].items()}
        collectibles = {}
        resource_cells = set()
        for x, y, kind in data["collectibles"]:
            collectibles[(x, y)] = collectibles.get((x, y), 0.0) + _KIND_VALUE[kind]
            if kind == "resource":
                resource_cells.add((x, y))
        instance = cls(int(data["grid_size"]), wall_between, bases, spawns, collectibles)
        instance.resource_cells = resource_cells
        return instance

    def identify_team(self, base_location):
        """Set self.team / our_base / enemy_bases from the observed base location.

        Exact match on the 6 known base coords; falls back to the nearest base
        (covers a non-novice eval where the location does not match).
        """
        loc = tuple(int(c) for c in base_location)
        team = next((t for t, b in self.bases.items() if b == loc), None)
        if team is None:
            team = min(self.bases,
                       key=lambda t: abs(self.bases[t][0] - loc[0])
                       + abs(self.bases[t][1] - loc[1]))
        self.team = team
        self.our_base = self.bases[team]
        self.enemy_bases = [b for t, b in self.bases.items() if t != team]
        return team
