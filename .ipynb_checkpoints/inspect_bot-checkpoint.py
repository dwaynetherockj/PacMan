"""
Log one full game at a chosen skill and summarise what the bot did, to see
why a given skill level behaves the way it does.

Run from the repo root:
    python -u inspect_bot.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
from collections import Counter

import pygame
pygame.time.wait = lambda ms: 0

from pacman import Game
from rl_env.bots import SkillBot

SKILL = 1.0
SEED = 0
MAX_TICKS = 10000


def main():
    random.seed(SEED)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()
    bot = SkillBot(SKILL, seed=SEED)
    board = engine.player.board
    tw, th = engine.tile_width, engine.tile_height

    reversals = 0
    prev_dir = None
    tile_visits = Counter()
    near_ghost_ticks = 0        # ticks spent within ~1 tile of a dangerous ghost
    lives_seen = engine.player.lives
    life_start = 0
    life_summaries = []

    for t in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()

        d = engine.player.direction
        if prev_dir is not None and d.name != prev_dir:
            # count only true reversals (opposite direction)
            opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            if opp.get(prev_dir) == d.name:
                reversals += 1
        prev_dir = d.name

        tile = (engine.player.location_y // th, engine.player.location_x // tw)
        tile_visits[tile] += 1

        for g in engine.ghosts:
            if g.is_frightened() or g.is_eaten():
                continue
            if (abs(g.location_x - engine.player.location_x) < tw and
                    abs(g.location_y - engine.player.location_y) < th):
                near_ghost_ticks += 1
                break

        if engine.player.lives < lives_seen:
            life_summaries.append((t - life_start, engine.level.score))
            lives_seen = engine.player.lives
            life_start = t

        if engine.game_over:
            break

    most_visited = tile_visits.most_common(3)
    print(f"Skill {SKILL}, seed {SEED}")
    print(f"Ticks played: {t}   final score: {engine.level.score}   "
          f"board cleared: {bool(((board==1)|(board==2)).sum()==0)}")
    print(f"Direction reversals: {reversals}")
    print(f"Ticks spent right next to a dangerous ghost: {near_ghost_ticks}")
    print(f"Most-revisited tiles (row,col): count -> {most_visited}")
    print(f"Per-life (ticks lasted, score at death): {life_summaries}")


if __name__ == "__main__":
    main()