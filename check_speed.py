"""Measure Blinky's real speed and check his picture stays on him."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pacman import Game

TICKS = 200

for mult in [1.0, 0.75, 1.25]:
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()          # skip the READY countdown
    ghost = engine.ghosts[0]              # Blinky
    for g in engine.ghosts:
        g.speed_mult = mult
    gap_start = (ghost.location_x - ghost.top_left_x,
                 ghost.location_y - ghost.top_left_y)
    moved = 0
    for _ in range(TICKS):
        x, y = ghost.location_x, ghost.location_y
        engine.tick()
        moved += abs(ghost.location_x - x) + abs(ghost.location_y - y)
    gap_end = (ghost.location_x - ghost.top_left_x,
               ghost.location_y - ghost.top_left_y)
    drift = (gap_end[0] - gap_start[0], gap_end[1] - gap_start[1])
    print(f"dial {mult}: {moved / TICKS:.2f} pixels per tick | picture drift {drift}")