"""
Test the deployed adaptive game (Plan B): original ghosts + speed dial,
driven by the controller. Runs weak and strong bot-players through full
games and checks the controller sets the right difficulty and it affects
how often the player is caught.

Run from the repo root:
    python -u test_plan_b.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
import pygame
pygame.time.wait = lambda ms: 0

from pacman import Game
from rl_env.bots import SkillBot
from rl_env.controller import AdaptiveController, BAND_SETTINGS

PLAYERS = {"weak": 0.0, "strong": 1.0}
GAMES = 10
MAX_TICKS = 8000


def play(player_skill, seed):
    random.seed(seed)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()
    bot = SkillBot(player_skill, seed=seed)
    controller = AdaptiveController(engine)

    start_lives = engine.player.lives
    for t in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()
        controller.update()
        if engine.game_over:
            break

    bands = [b for _, b in controller.history]
    return {"guesses": bands, "final_ghost_speed":
            BAND_SETTINGS[controller.current_band]["ghost_speed_mult"]}


def main():
    print(f"{GAMES} games per player type.\n")
    for pname, pskill in PLAYERS.items():
        all_bands = []
        for s in range(GAMES):
            r = play(pskill, seed=s)
            all_bands += r["guesses"]
        # count how often each band was chosen
        counts = {b: all_bands.count(b) for b in ["casual", "middle", "hardcore"]}
        total = sum(counts.values()) or 1
        print(f"{pname} player -> difficulty chosen across all lives: {counts}")
        print(f"   (casual {100*counts['casual']//total}%, "
              f"middle {100*counts['middle']//total}%, "
              f"hardcore {100*counts['hardcore']//total}%)\n")


if __name__ == "__main__":
    main()