"""
play.py - Play the AE Bomberman env yourself against 5 real agents.

Roster
------
The agent in each of the 6 AEC slots is chosen by the SLOT_AGENTS list
near the top of this file. Exactly one entry must be "human" — that is
your slot. The other five entries can be any of:

    "random"            — RandomAgent (uniform over legal actions)
    "greedy"            — GreedyAgent (A* to nearest live enemy base)
    "scripted:<name>"   — a named scripted strategy (see
                          ae/src/scripted/strategies.py for the list)
    "ckpt:<path>"       — a trained SymbolicTransformerActor at <path>

Edit SLOT_AGENTS and re-run; there are no CLI flags for the roster.

Controls
--------
    W           — FORWARD
    S           — BACKWARD
    A           — turn LEFT
    D           — turn RIGHT
    SPACE       — STAY
    B / F       — PLACE_BOMB
    R           — reset the environment (all agents also reset)
    T           — toggle respawn-timer overlay
    Q / ESC     — quit

    LEFT CLICK on anything — print entity info to terminal

Pass --verbose / -v to print action masks and per-step reward breakdowns.
"""

import os
import sys

# Resolve agent adapters and scripted strategies from the sibling ae/src tree.
# evaluate.py (RandomAgent/GreedyAgent/ScriptedAgent/NeuralAgent) lives in
# ae/src/training; scripted strategies live in ae/src. Mirror the trick from
# ae/src/training/visualize.py so `from evaluate import ...` and
# `from scripted.strategies import STRATEGIES` resolve when running play.py.
_PLAY_DIR = os.path.dirname(os.path.abspath(__file__))
_AE_SRC = os.path.abspath(os.path.join(_PLAY_DIR, "..", "ae", "src"))
_AE_TRAINING = os.path.join(_AE_SRC, "training")
for _p in (_AE_TRAINING, _AE_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import random

import pygame

from til_environment.actions import Action
from til_environment.config import default_config, load_config
from til_environment.entities import Agent, Bomb, Resource
from til_environment.bomberman_env import Bomberman

# Belief / MapPrior power the structure-redraw underneath the fog: the prior
# gives us walls and base positions deterministically up-front; the belief
# folds in per-turn observations so wall destruction and base kills are
# reflected in the player's view. Resolved via the sys.path bootstrap above.
from scripted.belief import Belief
from scripted.map_prior import MapPrior


KEY_TO_AGENT_ACTION = {
    pygame.K_w: Action.FORWARD,
    pygame.K_s: Action.BACKWARD,
    pygame.K_a: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.STAY,
    pygame.K_b: Action.PLACE_BOMB,
    pygame.K_f: Action.PLACE_BOMB,
}

FILLER_ACTION = int(Action.STAY)

# ---------------------------------------------------------------------------
# Roster lever. Edit this list to change who plays each AEC slot.
# Length must be 6. Exactly one entry must be "human" — that is your slot.
# Other entries can be:
#   "random"            — RandomAgent (uniform over legal actions)
#   "greedy"            — GreedyAgent (A* to nearest live enemy base)
#   "scripted:<name>"   — ScriptedAgent(<name>); see ae/src/scripted/strategies.py
#   "ckpt:<path>"       — NeuralAgent loading SymbolicTransformerActor from <path>
# ---------------------------------------------------------------------------
SLOT_AGENTS = [
    "scripted:balanced_extreme_opening",
    "human",
    "scripted:balanced_extreme_opening",
    "scripted:balanced_extreme_opening",
    "scripted:balanced_extreme_opening",
    "scripted:balanced_extreme_opening",
]


def _entity_info(entity) -> str:
    lines = [
        f"[{type(entity).__name__}] id={entity.entity_id}",
        f"  team={entity.team}  pos={entity.position.tolist()}",
    ]
    if isinstance(entity, Agent):
        lines += [
            f"  health={entity.health:.0f}/{entity.max_health:.0f}",
            f"  direction={entity.direction}",
            f"  frozen_ticks={entity.frozen_ticks}",
        ]
    elif isinstance(entity, Bomb):
        lines += [
            f"  timer={entity.timer}  attack={entity.attack}  "
            f"blast_radius={entity.blast_radius}",
            f"  placed_by={entity.attribute_rewards}",
        ]
    elif isinstance(entity, Resource):
        lines.append(f"  amount={entity.amount}")
    elif hasattr(entity, "health"):
        lines.append(f"  health={entity.health:.0f}/{entity.max_health:.0f}")
    if hasattr(entity, "reward_value"):
        lines.append(f"  reward_value={entity.reward_value}")
    return "\n".join(lines)


def _print_action_mask(env: Bomberman, agent_id: str) -> None:
    obs = env.observe(agent_id)
    mask = obs.get("action_mask")
    if mask is None:
        return
    available = [a.name for a in Action if mask[a]]
    blocked = [a.name for a in Action if not mask[a]]
    print(f"  ✓ {', '.join(available)}")
    if blocked:
        print(f"  ✗ {', '.join(blocked)}")
    print(
        f"  team_bombs={env.dynamics.team_bombs[env.dynamics.registry.get(agent_id).team]}  "
        f"team_resources={env.dynamics.team_resources[env.dynamics.registry.get(agent_id).team]:.2f}"
    )


def _print_scoreboard(env, slot_agents) -> None:
    """Print a sorted end-of-episode scoreboard with slot labels and a winner tag.

    Score = `env.dynamics.rewards.episode_rewards()[slot]`, the cumulative
    per-agent total for the episode. `env.rewards` alone is only the last
    step's reward and produced misleading 0.00 scores for slots that died
    earlier in the episode.
    """
    totals = env.dynamics.rewards.episode_rewards()
    rows = sorted(
        ((slot, slot_agents[slot][1], float(totals.get(slot, 0.0)))
         for slot in env.possible_agents),
        key=lambda r: r[2],
        reverse=True,
    )
    slot_w = max(len(slot) for slot, _, _ in rows)
    label_w = max(len(label) for _, label, _ in rows)
    score_w = max(len(f"{score:+.2f}") for _, _, score in rows)
    width = 5 + slot_w + 2 + label_w + 2 + score_w + len("  (winner)")

    print()
    title = " FINAL SCOREBOARD "
    pad = max(2, width - len(title) - 2)
    print(f"──{title}{'─' * pad}")
    print(f"  #  {'slot':<{slot_w}}  {'agent':<{label_w}}  {'score':>{score_w}}")
    for rank, (slot, label, score) in enumerate(rows, start=1):
        tag = "  (winner)" if rank == 1 else ""
        print(f"  {rank}  {slot:<{slot_w}}  {label:<{label_w}}  {score:+{score_w}.2f}{tag}")
    print("─" * width)
    print("Press R to reset or Q to quit.\n")


# Fog-of-war colours.
FOG_FILL = (24, 24, 30)                       # opaque dark grey covering non-LOS cells
INDESTRUCTIBLE_WALL_COLOUR = (190, 190, 195)  # bright grey — concrete
DESTRUCTIBLE_WALL_COLOUR = (200, 145, 80)     # warm brown — breakable
LIVE_ENEMY_BASE_COLOUR = (40, 200, 80)
DEAD_BASE_COLOUR = (220, 60, 60)
TEXT_COLOUR = (235, 235, 240)
TEXT_BG = (0, 0, 0, 190)

_HUD_FONT: pygame.font.Font | None = None


def _hud_font() -> pygame.font.Font:
    """Lazily build the HUD font (pygame must be initialised before this)."""
    global _HUD_FONT
    if _HUD_FONT is None:
        try:
            _HUD_FONT = pygame.font.Font("freesansbold.ttf", 14)
        except Exception:
            _HUD_FONT = pygame.font.SysFont(None, 18)
    return _HUD_FONT


def _render_player_view(env, player_slot, belief, frame, window) -> None:
    """Apply opaque fog-of-war + redraw deterministically-known structures + HUD.

    The env is configured with render_mode="rgb_array", so env.render(...)
    returns a numpy frame and writes the full-info frame to the MP4 sidecar
    (cfg.renderer.replay_dir) — but it does NOT touch any pygame display
    window. That prevents the un-fogged ground truth from flashing on
    screen between frames.

    Composition steps (all on `window`, then a single display.flip):
      1. Blit the env's just-drawn surface as the base layer. We pull
         from env.renderer._window directly (the Surface the env drew on)
         to avoid a numpy round-trip via `frame`.
      2. Paint OPAQUE fog over every cell outside the player team's
         current LOS (agent viewcone + own base viewcone via the env's
         `get_team_visible_area`). Enemies / bombs in fogged cells are
         genuinely hidden, not just shaded.
      3. Re-draw the structures the player is allowed to know from the
         deterministic map prior, ON TOP of the fog so they remain
         visible even in fogged cells:
           - intact walls: indestructible vs destructible get distinct
             colours so the player can plan around breakables. Uses
             `belief.is_wall` to drop walls observed destroyed.
           - enemy bases: green outline = alive, red X = killed (per
             `belief.dead_bases`).
      4. HUD overlay (top-left): bombs / resources / hp counters from
         the belief.

    Known leak (accepted trade-off): the env's full frame is the base
    layer, so any rendering the env paints on LOS cells — including the
    orange danger overlay for bombs OUTSIDE the player's LOS whose blasts
    reach INTO LOS cells — flows through. If you need strict no-cheat,
    switch back to the from-scratch belief render.
    """
    env_surface = env.renderer._window
    if env_surface is None or window is None:
        return

    pix = env.renderer.window_size / env.renderer.grid_size
    pix_i = int(pix)
    game_size = env.renderer.window_size

    # 1. Base layer: blit the env's freshly-drawn surface, cropped to the
    #    game grid in case the env has a debug panel.
    window.blit(env_surface, (0, 0),
                area=pygame.Rect(0, 0, game_size, game_size))

    if belief.prior is None:
        pygame.display.flip()
        return

    grid_size = belief.prior.grid_size

    # 2. Opaque fog over every cell the player can't currently see.
    player = env.dynamics.registry.get(player_slot)
    visible = set()
    if player is not None:
        visible = env.dynamics.vision.get_team_visible_area(
            team=player.team,
            registry=env.dynamics.registry,
            walls=env.dynamics.arena_state.walls,
            state=env.dynamics.arena_state._state,
        )
    fog_cell = pygame.Surface((pix_i + 1, pix_i + 1))
    fog_cell.fill(FOG_FILL)
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) in visible:
                continue
            window.blit(fog_cell, (int(x * pix), int(y * pix)))

    # 3a. Re-draw intact walls on top of fog, with destructible distinction.
    for pair in belief.prior.wall_between:
        a, b = tuple(pair)
        if not belief.is_wall(a, b):
            continue
        colour = (DESTRUCTIBLE_WALL_COLOUR if belief.is_destructible(a, b)
                  else INDESTRUCTIBLE_WALL_COLOUR)
        if a[0] == b[0]:                                  # horizontal edge
            top = a if a[1] < b[1] else b
            x0 = int(top[0] * pix)
            y0 = int((top[1] + 1) * pix)
            pygame.draw.line(window, colour, (x0, y0), (x0 + pix_i, y0), 3)
        else:                                              # vertical edge
            left = a if a[0] < b[0] else b
            x0 = int((left[0] + 1) * pix)
            y0 = int(left[1] * pix)
            pygame.draw.line(window, colour, (x0, y0), (x0, y0 + pix_i), 3)

    # 3b. Re-draw enemy bases on fogged cells (own base is in LOS so the
    #     env's draw of it already shows through).
    for base in belief.prior.enemy_bases:
        if base in visible:
            continue
        bx, by = base
        x0, y0 = int(bx * pix), int(by * pix)
        if base in belief.dead_bases:
            pygame.draw.line(window, DEAD_BASE_COLOUR,
                             (x0 + 3, y0 + 3),
                             (x0 + pix_i - 3, y0 + pix_i - 3), 3)
            pygame.draw.line(window, DEAD_BASE_COLOUR,
                             (x0 + pix_i - 3, y0 + 3),
                             (x0 + 3, y0 + pix_i - 3), 3)
        else:
            pygame.draw.rect(window, LIVE_ENEMY_BASE_COLOUR,
                             (x0 + 2, y0 + 2, pix_i - 4, pix_i - 4), 3)

    # 4. HUD: bomb / resource / hp counters in the top-left.
    font = _hud_font()
    hud_text = (f"bombs: {int(belief.team_bombs)}    "
                f"resources: {float(belief.team_resources):.2f}    "
                f"hp: {float(belief.health):.0f}")
    label = font.render(hud_text, True, TEXT_COLOUR)
    bg = pygame.Surface((label.get_width() + 14, label.get_height() + 8),
                        pygame.SRCALPHA)
    bg.fill(TEXT_BG)
    window.blit(bg, (6, 6))
    window.blit(label, (13, 10))

    pygame.display.flip()


def _handle_click_info(env, mouse_pos) -> None:
    """Left-click hit-test: print the clicked entity's info to the terminal.

    The earlier version of this helper also let the user pick a new slot
    to control mid-game; that path was removed when the roster became a
    fixed top-of-file SLOT_AGENTS constant.
    """
    entity = env.renderer.hit_test(mouse_pos[0], mouse_pos[1], env.dynamics.registry)
    if entity is None:
        print("[click] empty tile")
        return
    print("[click]")
    print(_entity_info(entity))


def _spec_to_agent(spec):
    """Parse a per-slot spec string into an (agent, label) pair.

    Specs:
      - "human"           -> (None, "human")  sentinel; the keyboard slot.
      - "random"          -> RandomAgent(), label "random"
      - "greedy"          -> GreedyAgent(), label "greedy"
      - "scripted:<name>" -> ScriptedAgent(<name>), label "scripted:<name>"
      - "ckpt:<path>"     -> NeuralAgent(SymbolicTransformerActor loaded from <path>),
                             label "ckpt:<basename>"
    Anything else raises ValueError.

    Kept in sync with ae/src/training/visualize.py:_spec_to_agent, plus the
    "human" sentinel that play.py needs and visualize.py does not.
    """
    if spec == "human":
        return None, "human"
    if spec == "random":
        from evaluate import RandomAgent
        return RandomAgent(), "random"
    if spec == "greedy":
        from evaluate import GreedyAgent
        return GreedyAgent(), "greedy"
    if spec.startswith("scripted:"):
        from evaluate import ScriptedAgent
        from scripted.strategies import STRATEGIES
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
        from evaluate import NeuralAgent
        from policy import SymbolicTransformerActor
        actor = SymbolicTransformerActor.from_checkpoint(path)
        actor.eval()
        label = f"ckpt:{os.path.basename(path)}"
        return NeuralAgent(actor, name=label), label
    raise ValueError(
        f"bad agent spec {spec!r}; expected 'human', 'random', 'greedy', "
        f"'scripted:<name>', or 'ckpt:<path>'")


def _validate_slot_agents(specs):
    """Raise ValueError unless `specs` is a length-6 list with exactly one 'human'.

    `specs` is the SLOT_AGENTS roster: one spec string per AEC slot
    (agent_0..agent_5). Exactly one entry must be the literal "human" — that
    is the slot the keyboard player drives. The rest can be "random",
    "greedy", "scripted:<name>", or "ckpt:<path>" (parsed by _spec_to_agent).
    """
    if len(specs) != 6:
        raise ValueError(
            f"SLOT_AGENTS must have length 6 (one entry per AEC slot); "
            f"got length {len(specs)}: {specs!r}")
    human_count = sum(1 for s in specs if s == "human")
    if human_count != 1:
        raise ValueError(
            f"SLOT_AGENTS must contain exactly one 'human' entry "
            f"(got {human_count}): {specs!r}")


def _build_slot_agents(specs, possible_agents):
    """Build the slot-id -> (agent, label) map and pin the human's slot id.

    Validates `specs` first (length 6, exactly one "human"). Then maps each
    spec to an (agent, label) via `_spec_to_agent`. The human slot stores
    (None, "human"); bot slots store concrete agent instances.

    Returns (slot_agents, player_slot) where slot_agents is a dict keyed by
    possible_agents[i] (e.g. "agent_0") and player_slot is the slot id the
    human controls. Raises ValueError on a bad roster.
    """
    _validate_slot_agents(specs)
    if len(possible_agents) != len(specs):
        raise ValueError(
            f"possible_agents length {len(possible_agents)} "
            f"!= specs length {len(specs)}")
    slot_agents = {}
    player_slot = None
    for slot, spec in zip(possible_agents, specs):
        agent, label = _spec_to_agent(spec)
        slot_agents[slot] = (agent, label)
        if spec == "human":
            player_slot = slot
    return slot_agents, player_slot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (defaults to til_environment/bomberman_config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print action masks and per-step reward breakdowns.",
    )
    args = parser.parse_args()
    verbose: bool = args.verbose

    if args.config is None:
        cfg = default_config()
    else:
        cfg = load_config(args.config)

    cfg.env.novice = True
    # rgb_array (not "human") so the env never calls pygame.display.update()
    # itself — otherwise the un-fogged ground-truth frame flashes on screen
    # between our fog overlays. The env still composes its full-info frame
    # on env.renderer._window (an in-memory Surface) which we then blit
    # into our own window in _render_player_view.
    cfg.env.render_mode = "rgb_array"
    # Persist the full-information game frames to MP4 (one file per episode)
    # so we can review the actual game after playing with fog of war.
    cfg.renderer.replay_dir = os.path.join(_PLAY_DIR, "logs", "play_replays")

    env = Bomberman(cfg)
    env.reset(seed=random.randint(0, 99999))

    # Open OUR window — env doesn't open one in rgb_array mode.
    pygame.init()
    window = pygame.display.set_mode((env.renderer.window_size, env.renderer.window_size))
    pygame.display.set_caption("TIL-AI Bomberman (fog of war)")

    # Fog-of-war state. prior is reused across episodes (novice map is
    # deterministic); belief is reset per episode so wall-destruction and
    # base-kill memory doesn't bleed between games.
    prior = MapPrior.load()
    belief = Belief()
    needs_belief_init = True       # flips False after the first observation
                                   # of an episode is folded in (and again
                                   # after every R-reset).

    slot_agents, player_slot = _build_slot_agents(SLOT_AGENTS, env.possible_agents)
    for agent, _label in slot_agents.values():
        if agent is not None:
            agent.reset()
    print(f"Controlling: {player_slot}")
    for slot in env.possible_agents:
        print(f"  {slot}  -> {slot_agents[slot][1]}")

    running = True
    clock = pygame.time.Clock()
    show_respawn_overlay = False

    while running:
        agent = env.agent_selection

        is_new_round = env.agent_selector.is_first()
        if is_new_round:
            obs = env.observe(player_slot)
            if needs_belief_init:
                if prior.our_base is None:
                    prior.identify_team(obs["base_location"])
                belief.reset(prior)
                needs_belief_init = False
            belief.update(obs)

            _overlay = env.dynamics.respawn_map if show_respawn_overlay else None
            frame = env.render(selected_agent_id=player_slot, respawn_overlay=_overlay)
            _render_player_view(env, player_slot, belief, frame, window)

            clock.tick(env.cfg.renderer.render_fps)

        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                _print_scoreboard(env, slot_agents)
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                running = False
                                waiting = False
                            elif event.key == pygame.K_r:
                                env.reset(seed=random.randint(0, 99999))
                                belief = Belief()
                                needs_belief_init = True
                                for a, _label in slot_agents.values():
                                    if a is not None:
                                        a.reset()
                                waiting = False
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            _handle_click_info(env, event.pos)
            continue

        if agent == player_slot:
            if verbose:
                _print_action_mask(env, agent)
            agent_action = None
            while agent_action is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        agent_action = Action.STAY
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                            agent_action = Action.STAY
                        elif event.key == pygame.K_r:
                            env.reset(seed=random.randint(0, 99999))
                            belief = Belief()
                            needs_belief_init = True
                            for a, _label in slot_agents.values():
                                if a is not None:
                                    a.reset()
                            agent_action = None
                            break
                        elif event.key == pygame.K_t:
                            show_respawn_overlay = not show_respawn_overlay
                            print(f"[respawn overlay] {'ON' if show_respawn_overlay else 'OFF'}")
                        elif event.key in KEY_TO_AGENT_ACTION:
                            agent_action = KEY_TO_AGENT_ACTION[event.key]
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        _handle_click_info(env, event.pos)
                        _overlay = env.dynamics.respawn_map if show_respawn_overlay else None
                        frame = env.render(selected_agent_id=player_slot, respawn_overlay=_overlay)
                        _render_player_view(env, player_slot, belief, frame, window)
            if not running:
                break
            env.step(int(agent_action))
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset(seed=random.randint(0, 99999))
                        belief = Belief()
                        needs_belief_init = True
                        for a, _label in slot_agents.values():
                            if a is not None:
                                a.reset()
                    elif event.key == pygame.K_t:
                        show_respawn_overlay = not show_respawn_overlay
                        print(f"[respawn overlay] {'ON' if show_respawn_overlay else 'OFF'}")
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    _handle_click_info(env, event.pos)
            obs = env.observe(agent)
            env.step(int(slot_agents[agent][0].action(obs)))

        if verbose:
            print(
                "step rewards:",
                {a: f"{r:.2f}" for a, r in env.dynamics.rewards._step.items()},
            )
            print(
                "ep rewards:",
                {a: f"{r:.2f}" for a, r in env.dynamics.rewards._episode.items()},
            )
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
