"""Episode-replay visualizer for AE self-play training.

Renders one labelled MP4 of an episode: each AEC slot agent_0..agent_5 is
drawn by the env, and a PIL legend band naming the policy controlling each
slot is stacked above every frame. Used both as a standalone CLI and by
train_selfplay.py's periodic viz hook. The vendored env is NOT modified.
"""
import os
import sys

# A plain `uv run python visualize.py` only has ae/training on sys.path; add
# ae/src so `from policy import ...` / `from features import ...` resolve.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from evaluate import GreedyAgent, NeuralAgent, RandomAgent, ScriptedAgent
from policy import SymbolicTransformerActor
from scripted.strategies import STRATEGIES

import torch


def _spec_to_agent(spec):
    """Parse a per-slot spec string into an (agent, label) pair.

    Specs:
      - "random"          -> RandomAgent(), label "random"
      - "scripted:<name>" -> ScriptedAgent(<name>), label "scripted:<name>"
      - "ckpt:<path>"     -> NeuralAgent(SymbolicTransformerActor loaded from <path>),
                             label "ckpt:<basename>"
    Anything else raises ValueError.
    """
    if spec == "random":
        return RandomAgent(), "random"
    if spec == "greedy":
        return GreedyAgent(), "greedy"
    if spec.startswith("scripted:"):
        name = spec[len("scripted:"):]
        if name not in STRATEGIES:
            raise ValueError(
                f"unknown scripted strategy {name!r}; "
                f"valid: {sorted(STRATEGIES)}")
        return ScriptedAgent(name), f"scripted:{name}"
    if spec.startswith("ckpt:"):
        path = spec[len("ckpt:"):]
        if not os.path.exists(path):
            raise ValueError(f"checkpoint not found: {path}")
        actor = SymbolicTransformerActor.from_checkpoint(path)
        actor.eval()
        label = f"ckpt:{os.path.basename(path)}"
        return NeuralAgent(actor, name=label), label
    raise ValueError(
        f"bad agent spec {spec!r}; expected 'random', "
        f"'scripted:<name>', or 'ckpt:<path>'")


from PIL import Image, ImageDraw

from til_environment.renderer import _team_color

# Legend layout knobs.
LEGEND_ROW_H = 28          # KNOB: pixel height of one legend row
_SWATCH_PAD = 6            # inset of the color swatch within its row
_TEXT_X = LEGEND_ROW_H     # text starts just right of the swatch column


def _norm_color(c):
    """Normalize a _team_color() result to a (r, g, b) tuple of 0-255 ints.

    _team_color returns 0-255 ints for the fixed first-four-team palette and
    0-1 floats for golden-ratio-generated teams; normalize both to ints.
    """
    r, g, b = c
    if max(r, g, b) <= 1.0 and isinstance(r, float):
        return (int(r * 255), int(g * 255), int(b * 255))
    return (int(r), int(g), int(b))


def _build_legend(labels, width, layers=None):
    """Render the slot->policy legend as a uint8 RGB array (H, width, 3).

    One LEGEND_ROW_H-tall row per label: a team-color swatch (slot agent_K is
    team K in the novice 6-team layout) followed by 'agent_K = <label>'. When
    `layers` is given (one string per label), the agent's current cascade
    layer is appended to its row in brackets.
    """
    if layers is not None and len(layers) != len(labels):
        raise ValueError(
            f"layers length {len(layers)} != labels length {len(labels)}")
    height = LEGEND_ROW_H * len(labels)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for k, label in enumerate(labels):
        y0 = k * LEGEND_ROW_H
        color = _norm_color(_team_color(k))
        # color swatch
        draw.rectangle(
            [_SWATCH_PAD, y0 + _SWATCH_PAD,
             LEGEND_ROW_H - _SWATCH_PAD, y0 + LEGEND_ROW_H - _SWATCH_PAD],
            fill=color, outline=(0, 0, 0))
        # label text (default PIL bitmap font — no font file dependency)
        text = f"agent_{k} = {label}"
        if layers is not None and layers[k]:
            text += f"    [{layers[k]}]"
        draw.text((_TEXT_X, y0 + _SWATCH_PAD), text, fill=(0, 0, 0))
        # thin separator line under the row
        draw.line([0, y0 + LEGEND_ROW_H - 1, width, y0 + LEGEND_ROW_H - 1],
                  fill=(211, 211, 211))
    return np.asarray(img, dtype=np.uint8)


def _agent_layer(agent):
    """The agent's current cascade layer, or '' for agents with no cascade.

    Scripted agents expose it via `agent.belief.last_layer`; RandomAgent /
    NeuralAgent have no belief, so they resolve to ''."""
    belief = getattr(agent, "belief", None)
    return getattr(belief, "last_layer", None) or ""


# Belief panel layout knobs.
BELIEF_CELL_PX = 24       # KNOB: pixel size of one cell in the belief grid
BELIEF_PAD = 8            # margin between the grid and the panel border

_BG = (255, 255, 255)
_GRID = (215, 215, 215)
_WALL_INTACT = (40, 40, 40)
_WALL_GONE = (220, 220, 220)
_DEAD_BASE = (255, 80, 80)
_LIVE_BASE = (60, 180, 75)
_RESOURCE = (250, 215, 60)         # remaining resource-kind collectible
_NON_RESOURCE_COLLECTIBLE = (220, 220, 160)   # remaining recon/mission
_ENEMY_BOMB = (220, 30, 30)
_FROZEN_ENEMY = (140, 140, 200)
_LIVE_ENEMY = (200, 60, 200)
_AGENT = (40, 100, 230)
_VISIBLE = (220, 240, 255)         # background tint for currently-visible cells


def _visible_cells(belief):
    """Set of (x, y) cells the LAST `update()` actually saw via LOS.

    `belief.last_visible_cells` is repopulated each `update()` from cells
    whose tile-type channel was set (= cells the env's LOS marked visible).
    Includes both the agent's directional viewcone AND the base viewcone
    (folded by `update()` whenever our base is alive). LOS-occluded cells
    inside the 7×5/7×7 bounding rectangles are correctly excluded.
    """
    return set(getattr(belief, "last_visible_cells", ()) or ())


def _facing_arrow(cell_xy, facing, cell_px):
    """Return (cx, cy, ex, ey) for a small arrow from the cell centre toward facing."""
    cx = cell_xy[0] + cell_px // 2
    cy = cell_xy[1] + cell_px // 2
    half = cell_px // 2 - 3
    if facing == 0:    # RIGHT
        return (cx, cy, cx + half, cy)
    if facing == 1:    # DOWN
        return (cx, cy, cx, cy + half)
    if facing == 2:    # LEFT
        return (cx, cy, cx - half, cy)
    return (cx, cy, cx, cy - half)  # UP


def _render_belief(belief, width):
    """Render the scripted agent's belief as a side panel of width `width`.

    Returns a uint8 RGB array of shape (height, width, 3) where height is
    chosen to fit a square grid of belief.prior.grid_size cells.
    Returns None when there's no belief to render (e.g. agent_0 isn't scripted).
    """
    if belief is None or belief.prior is None or belief.location is None:
        return None
    gs = belief.prior.grid_size
    cell_px = max(8, min(BELIEF_CELL_PX, (width - 2 * BELIEF_PAD) // gs))
    grid_px = cell_px * gs
    # Header: two lines (state + counters), then the grid.
    title_h = 44
    height = title_h + grid_px + 2 * BELIEF_PAD
    img = Image.new("RGB", (width, height), color=_BG)
    draw = ImageDraw.Draw(img)
    x0 = BELIEF_PAD + (width - 2 * BELIEF_PAD - grid_px) // 2
    y0 = title_h + BELIEF_PAD

    layer = belief.last_layer or ""
    draw.text((BELIEF_PAD, 4),
              f"belief  step={belief.step}  layer=[{layer}]",
              fill=(0, 0, 0))
    draw.text((BELIEF_PAD, 22),
              f"bombs: {belief.team_bombs}    "
              f"resources: {belief.team_resources:.3f}    "
              f"hp: {belief.health:.0f}",
              fill=(0, 0, 0))

    # Visible-cell tint (drawn first, so subsequent layers paint over it).
    for cell in _visible_cells(belief):
        cx, cy = x0 + cell[0] * cell_px, y0 + cell[1] * cell_px
        draw.rectangle([cx, cy, cx + cell_px, cy + cell_px], fill=_VISIBLE)

    # Cell grid lines.
    for i in range(gs + 1):
        draw.line([x0, y0 + i * cell_px, x0 + grid_px, y0 + i * cell_px], fill=_GRID)
        draw.line([x0 + i * cell_px, y0, x0 + i * cell_px, y0 + grid_px], fill=_GRID)

    remaining = belief.remaining_collectibles()
    resource_cells = getattr(belief.prior, "resource_cells", None) or set()
    enemy_base_set = set(getattr(belief.prior, "enemy_bases", []))
    dead_bases = belief.dead_bases
    live_enemies = belief.live_enemies()
    frozen_enemies = belief.frozen_enemies
    enemy_bombs = belief.enemy_bombs

    # Collectibles.
    for cell in remaining:
        cx, cy = x0 + cell[0] * cell_px, y0 + cell[1] * cell_px
        fill = _RESOURCE if cell in resource_cells else _NON_RESOURCE_COLLECTIBLE
        draw.rectangle([cx + 4, cy + 4, cx + cell_px - 4, cy + cell_px - 4],
                       fill=fill, outline=(0, 0, 0))

    # Enemy bases (live = green outline; dead = red X).
    for base in enemy_base_set:
        cx, cy = x0 + base[0] * cell_px, y0 + base[1] * cell_px
        if base in dead_bases:
            draw.line([cx + 3, cy + 3, cx + cell_px - 3, cy + cell_px - 3],
                      fill=_DEAD_BASE, width=2)
            draw.line([cx + cell_px - 3, cy + 3, cx + 3, cy + cell_px - 3],
                      fill=_DEAD_BASE, width=2)
        else:
            draw.rectangle([cx + 2, cy + 2, cx + cell_px - 2, cy + cell_px - 2],
                           outline=_LIVE_BASE, width=2)

    # Enemy bombs (red square + countdown text).
    for cell, timer in enemy_bombs.items():
        cx, cy = x0 + cell[0] * cell_px, y0 + cell[1] * cell_px
        draw.rectangle([cx + 5, cy + 5, cx + cell_px - 5, cy + cell_px - 5],
                       fill=_ENEMY_BOMB)
        draw.text((cx + cell_px // 2 - 3, cy + cell_px // 2 - 5), str(timer),
                  fill=(255, 255, 255))

    # Enemies.
    for cell in frozen_enemies:
        cx, cy = x0 + cell[0] * cell_px, y0 + cell[1] * cell_px
        draw.ellipse([cx + 6, cy + 6, cx + cell_px - 6, cy + cell_px - 6],
                     fill=_FROZEN_ENEMY)
    for cell in live_enemies:
        cx, cy = x0 + cell[0] * cell_px, y0 + cell[1] * cell_px
        draw.ellipse([cx + 4, cy + 4, cx + cell_px - 4, cy + cell_px - 4],
                     fill=_LIVE_ENEMY, outline=(0, 0, 0))

    # Walls — iterate prior pairs and check is_wall(a, b).
    for pair in belief.prior.wall_between:
        a, b = tuple(pair)
        # Determine the edge between a and b.
        if a[0] == b[0]:        # horizontal edge (different y)
            top = a if a[1] < b[1] else b
            x_px = x0 + top[0] * cell_px
            y_px = y0 + (top[1] + 1) * cell_px
            colour = _WALL_INTACT if belief.is_wall(a, b) else _WALL_GONE
            draw.line([x_px, y_px, x_px + cell_px, y_px], fill=colour, width=2)
        else:                   # vertical edge (different x)
            left = a if a[0] < b[0] else b
            x_px = x0 + (left[0] + 1) * cell_px
            y_px = y0 + left[1] * cell_px
            colour = _WALL_INTACT if belief.is_wall(a, b) else _WALL_GONE
            draw.line([x_px, y_px, x_px, y_px + cell_px], fill=colour, width=2)

    # Agent (blue filled circle + facing arrow).
    loc = belief.location
    cx, cy = x0 + loc[0] * cell_px, y0 + loc[1] * cell_px
    draw.ellipse([cx + 5, cy + 5, cx + cell_px - 5, cy + cell_px - 5],
                 fill=_AGENT)
    arrow = _facing_arrow((cx, cy), belief.facing, cell_px)
    draw.line(arrow, fill=(255, 255, 255), width=2)

    return np.asarray(img, dtype=np.uint8)


import random

import imageio

from til_environment import bomberman_env
from til_environment.config import default_config

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]


def render_episode(slot_agents, out_path, *, fps=5, max_steps=200, seed=0):
    """Run one labelled render episode and write it to out_path as an MP4.

    slot_agents: {"agent_0": (agent, label), ..., "agent_5": (agent, label)}
        agent has .reset() and .action(observation) (evaluate.py adapters).
    Returns {"path": out_path, "scores": {slot: total_reward}, "steps": n}.
    """
    cfg = default_config()
    cfg.env.novice = True
    cfg.env.render_mode = "rgb_array"
    env = bomberman_env.basic_env(cfg=cfg, env_wrappers=[])

    labels = [slot_agents[s][1] for s in SLOTS]
    # the legend is rebuilt only when an agent's live cascade layer changes
    current_layers = {s: "" for s in SLOTS}
    legend = None
    last_layers = None

    random.seed(seed)
    env.reset(seed=seed)
    for agent, _ in slot_agents.values():
        agent.reset()

    totals = {s: 0.0 for s in SLOTS}
    frames = []
    steps = 0
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        totals[slot] += float(reward)
        if term or trunc:
            env.step(None)
            current_layers[slot] = ""        # drop the stale layer of a dead slot
        else:
            agent = slot_agents[slot][0]
            env.step(agent.action(obs))
            current_layers[slot] = _agent_layer(agent)
        frame = env.render()
        if frame is not None:
            frame = np.asarray(frame, dtype=np.uint8)
            # Belief side panel for agent_0 (None when the slot isn't scripted).
            agent0_belief = getattr(slot_agents["agent_0"][0], "belief", None)
            panel = _render_belief(agent0_belief, width=frame.shape[1] // 2)
            if panel is not None:
                # Pad heights so hstack lines up.
                if panel.shape[0] < frame.shape[0]:
                    pad = np.full((frame.shape[0] - panel.shape[0],
                                   panel.shape[1], 3), 255, dtype=np.uint8)
                    panel = np.vstack([panel, pad])
                elif panel.shape[0] > frame.shape[0]:
                    pad = np.full((panel.shape[0] - frame.shape[0],
                                   frame.shape[1], 3), 255, dtype=np.uint8)
                    frame = np.vstack([frame, pad])
                composed = np.hstack([frame, panel])
            else:
                composed = frame
            new_layers = [current_layers[s] for s in SLOTS]
            if new_layers != last_layers:
                legend = _build_legend(labels, width=composed.shape[1],
                                       layers=new_layers)
                last_layers = new_layers
            frames.append(np.vstack([legend, composed]))
        steps += 1
        if steps >= max_steps * len(SLOTS):
            break
    env.close()

    if not frames:
        raise RuntimeError("render_episode captured no frames — is the env's render_mode set to 'rgb_array'?")
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264")
    # report episode length in env steps (agent_iter yields one turn per
    # slot, so a 6-agent env advances ~len(SLOTS) iters per env step).
    return {"path": out_path,
            "scores": totals,
            "steps": steps // len(SLOTS)}


from dataclasses import dataclass, field

VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz")


@dataclass
class VizArgs:
    """CLI args for a standalone replay render."""
    agents: list = field(default_factory=lambda: ["random"] * 6)  # 6 specs
    out: str = os.path.join(VIZ_DIR, "demo.mp4")
    fps: int = 5            # KNOB
    max_steps: int = 200    # KNOB
    seed: int = 0
    trace_walls: bool = False   # print belief wall-destruction log after render
    trace_decisions: bool = False  # print per-step cascade decision log
    trace_observations: bool = False  # print per-step per-cell wall observations
    trace_out: str = ""          # if set, write all enabled traces to this file


def run_cli(args):
    """Build a slot_agents dict from args.agents and render one episode."""
    if len(args.agents) != len(SLOTS):
        raise ValueError(
            f"--agents needs exactly {len(SLOTS)} specs, "
            f"got {len(args.agents)}")
    slot_agents = {}
    for slot, spec in zip(SLOTS, args.agents):
        agent, label = _spec_to_agent(spec)
        slot_agents[slot] = (agent, label)
        belief = getattr(agent, "belief", None)
        if belief is not None:
            if args.trace_walls:
                belief.trace_wall_destruction = True
            if args.trace_decisions:
                belief.trace_decisions = True
            if args.trace_observations:
                belief.trace_observations = True
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result = render_episode(slot_agents, args.out,
                            fps=args.fps, max_steps=args.max_steps,
                            seed=args.seed)
    # Render destination for the dump: file (args.trace_out) or stdout.
    if (args.trace_walls or args.trace_decisions or args.trace_observations):
        sink = open(args.trace_out, "w") if args.trace_out else None

        def _emit(line):
            print(line, file=sink) if sink else print(line)

        for slot, (agent, label) in slot_agents.items():
            belief = getattr(agent, "belief", None)
            if belief is None:
                continue
            if args.trace_walls:
                log = belief.wall_destruction_log
                _emit(f"\n[{slot} {label}] wall-destruction events ({len(log)}):")
                for ev in log:
                    _emit(f"  step {ev['step']:>3} agent={ev['agent_loc']} "
                          f"face={ev['agent_facing']}: "
                          f"cell {ev['cell']} {ev['channel']}={ev['channel_value']:.2f} "
                          f"-> pair {ev['cell']}--{ev['neighbor']}  "
                          f"(prior destructible={ev['wall_destructible_in_prior']}, "
                          f"DESTR_ch={ev['destructible_channel_value']})")
            if args.trace_decisions:
                log = belief.decision_log
                _emit(f"\n[{slot} {label}] cascade-decision trace ({len(log)} entries):")
                last_step = None
                for step, layer, key, value in log:
                    if step != last_step:
                        _emit(f"  ── step {step} ──")
                        last_step = step
                    _emit(f"    {layer:>14}: {key} = {value!r}")
            if args.trace_observations:
                log = belief.observation_log
                _emit(f"\n[{slot} {label}] per-cell wall observations ({len(log)} entries):")
                last_step = None
                for ev in log:
                    if ev["step"] != last_step:
                        _emit(f"  ── step {ev['step']}  agent={ev['agent_loc']} face={ev['agent_facing']} ──")
                        last_step = ev["step"]
                    _emit(f"    cell {ev['cell']} viewidx={ev['view_index']} "
                          f"R={ev['WALL_RIGHT']:.1f} D={ev['WALL_DOWN']:.1f} "
                          f"L={ev['WALL_LEFT']:.1f} U={ev['WALL_UP']:.1f} | "
                          f"dR={ev['DESTR_WALL_RIGHT']:.1f} dD={ev['DESTR_WALL_DOWN']:.1f} "
                          f"dL={ev['DESTR_WALL_LEFT']:.1f} dU={ev['DESTR_WALL_UP']:.1f}")
        if sink:
            sink.close()
            print(f"trace written to {args.trace_out}")
    return result


def main():
    import tyro
    args = tyro.cli(VizArgs)
    result = run_cli(args)
    print(f"wrote replay -> {result['path']} "
          f"({result['steps']} steps, scores={result['scores']})")


if __name__ == "__main__":
    main()
