"""Offline tool: dump the deterministic novice-mode arena to arena_map.json.

Run from the training env:  cd ae/training && uv run python ../tools/dump_map.py
The novice map is fixed (maze seed 19, scatter seed 88), so the output is
byte-identical on every run.
"""
import json
from pathlib import Path

import numpy as np

from til_environment import bomberman_env
from til_environment.config import default_config
from til_environment.entities import Mission, Recon, Resource

OUT = Path(__file__).resolve().parents[1] / "src" / "arena_map.json"
RESPAWN_OUT = Path(__file__).resolve().parents[1] / "src" / "respawn_map.json"


def main() -> None:
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=0)
    dyn = env.unwrapped.dynamics
    reg = dyn.registry
    arena = dyn.arena_state

    walls = sorted(
        [int(we.ax), int(we.ay), int(we.direction.value), bool(we.destructible)]
        for we in arena.wall_edges.values()
    )
    bases = {str(b.team): [int(b.position[0]), int(b.position[1])]
             for b in reg.bases()}
    spawns = {str(a.team): {"pos": [int(a.position[0]), int(a.position[1])],
                            "facing": int(a.direction)}
              for a in reg.agents()}
    collectibles = []
    for cls, kind in ((Mission, "mission"), (Recon, "recon"),
                      (Resource, "resource")):
        for e in reg.by_type(cls):
            collectibles.append([int(e.position[0]), int(e.position[1]), kind])
    collectibles.sort()

    data = {
        "grid_size": int(env.unwrapped.grid_size),
        "walls": walls,
        "bases": bases,
        "spawns": spawns,
        "collectibles": collectibles,
    }
    # The novice respawn_map is deterministic (Perlin-seeded with rng_seed=88
    # under novice mode). Dump it so features.py can maintain a per-tile
    # respawn-countdown channel offline with zero env access.
    respawn_map = np.asarray(dyn.respawn_map, dtype=int).tolist()
    RESPAWN_OUT.write_text(json.dumps(respawn_map))
    print(f"wrote {RESPAWN_OUT}  (16x16 respawn-delay grid)")
    env.close()
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"wrote {OUT}  ({len(walls)} walls, {len(collectibles)} collectibles, "
          f"{len(bases)} bases)")


if __name__ == "__main__":
    main()
