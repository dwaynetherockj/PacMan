"""
Play full games with the skill bots against the ORIGINAL ghosts and print
average results per skill level. Weak bots should do worse than strong ones.

Run from the repo root:
    python -u run_bot_games.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

import pygame
pygame.time.wait = lambda ms: 0  # skip the game's 500 ms freeze

from pacman import Game
from rl_env.bots import SkillBot

SKILLS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMES_PER_SKILL = 20
MAX_TICKS = 10000


def play(skill, seed):
    random.seed(seed)
    game = Game()
    engine = game.game_engine
    bot = SkillBot(skill, seed=seed)
    board = engine.player.board
    dots_at_start = int(((board == 1) | (board == 2)).sum())

    tick = 0
    for tick in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()
        if engine.game_over:
            break

    dots_left = int(((board == 1) | (board == 2)).sum())
    return {
        "ticks": tick,
        "score": engine.level.score,
        "lives_left": engine.player.lives,
        "dots_eaten": dots_at_start - dots_left,
        "dots_left": dots_left,
        "game_over": bool(engine.game_over),
        "board_cleared": dots_left == 0,
    }


def main():
    print(f"{GAMES_PER_SKILL} games per skill level, up to {MAX_TICKS} ticks, "
          f"original 4 ghosts, 3 lives")
    for skill in SKILLS:
        runs = [play(skill, seed=s) for s in range(GAMES_PER_SKILL)]
        avg = lambda key: sum(r[key] for r in runs) / len(runs)
        overs = sum(r["game_over"] for r in runs)
        cleared = sum(r["board_cleared"] for r in runs)
        timeouts = sum(1 for r in runs
                       if not r["game_over"] and not r["board_cleared"])
        print(f"skill {skill:.2f}: survived {avg('ticks'):6.0f} ticks, "
              f"score {avg('score'):6.0f}, dots {avg('dots_eaten'):5.1f}, "
              f"dots left {avg('dots_left'):5.1f}, lives left {avg('lives_left'):4.1f}, "
              f"game-overs {overs}/{len(runs)}, cleared {cleared}/{len(runs)}, "
              f"ran out of clock {timeouts}/{len(runs)}", flush=True)


if __name__ == "__main__":
    main()