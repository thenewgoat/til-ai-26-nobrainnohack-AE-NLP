"""Our channel indices must match til_environment.observation.ViewChannel."""
from til_environment.observation import ViewChannel

from scripted.belief import CH


def test_channel_indices_match_env():
    assert CH["TILE_EMPTY"] == ViewChannel.TILE_EMPTY
    assert CH["TILE_RECON"] == ViewChannel.TILE_RECON
    assert CH["TILE_MISSION"] == ViewChannel.TILE_MISSION
    assert CH["TILE_RESOURCE"] == ViewChannel.TILE_RESOURCE
    assert CH["WALL_RIGHT"] == ViewChannel.WALL_RIGHT
    assert CH["WALL_DOWN"] == ViewChannel.WALL_DOWN
    assert CH["WALL_LEFT"] == ViewChannel.WALL_LEFT
    assert CH["WALL_UP"] == ViewChannel.WALL_UP
    assert CH["ENEMY_AGENT"] == ViewChannel.ENEMY_AGENT
    assert CH["ENEMY_AGENT_HEALTH"] == ViewChannel.ENEMY_AGENT_HEALTH
    assert CH["ENEMY_BASE"] == ViewChannel.ENEMY_BASE
    assert CH["ENEMY_BASE_HEALTH"] == ViewChannel.ENEMY_BASE_HEALTH
    assert CH["ALLY_BOMB"] == ViewChannel.ALLY_BOMB
    assert CH["ENEMY_BOMB"] == ViewChannel.ENEMY_BOMB
    assert CH["ALLY_BOMB_TIMER"] == ViewChannel.ALLY_BOMB_TIMER
    assert CH["ENEMY_BOMB_TIMER"] == ViewChannel.ENEMY_BOMB_TIMER
