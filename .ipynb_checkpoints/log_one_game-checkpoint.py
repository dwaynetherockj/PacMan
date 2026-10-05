"""
Play one full game with a skill bot and log every tick to a CSV.

Run from the repo root:
    python -u log_one_game.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
import pygame
pygame.time.wait = lambda ms: 0

from pacman import Game
from rl_env.bots import SkillBot
from rl_env.logger import TelemetryLogger

SKILL = 0.5
SEED = 0
MAX_TICKS = 10000
OUT = "telemetry_sample.csv"


def main():
    random.seed(SEED)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()
    bot = SkillBot(SKILL, seed=SEED)
    logger = TelemetryLogger(engine, OUT)

    for _ in range(MAX_TICKS):
        engine.direction_command = bot.choose(engine)
        engine.tick()
        logger.log()
        if engine.game_over:
            break

    logger.close()
    print(f"Logged {logger.tick} ticks to {OUT}")
    print(f"Final score {engine.level.score}, lives {engine.player.lives}")


if __name__ == "__main__":
    main()