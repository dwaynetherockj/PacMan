"""
Watch the adaptive controller work.

Plays one game with a bot as the player. Each time the player loses a life,
the controller guesses the player's skill and sets the ghost difficulty for
the next life. This prints what it guessed and what difficulty it picked, so
we can see the whole loop working.

Run from the repo root:
    python -u test_controller.py
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

# Pick which kind of player to simulate: 0.0 = weak, 1.0 = strong.
PLAYER_SKILL = 0.75
SEED = 0
MAX_TICKS = 10000


def main():
    random.seed(SEED)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()
    bot = SkillBot(PLAYER_SKILL, seed=SEED)
    controller = AdaptiveController(engine)

    print(f"Simulated player skill: {PLAYER_SKILL} "
          f"(0.0 = weak, 1.0 = strong)\n")

    lives_seen = engine.player.lives
    for t in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()
        controller.update()

        # Print whenever a life was just lost and the controller reacted.
        if engine.player.lives < lives_seen:
            lives_seen = engine.player.lives
            band = controller.current_band
            mult = BAND_SETTINGS[band]["ghost_speed_mult"]
            print(f"Life lost. Controller guessed: {band:<9} "
                  f"-> ghost speed set to {mult}x")

        if engine.game_over:
            break

    print("\nFull sequence of guesses (life number, guess):")
    print(controller.history)


if __name__ == "__main__":
    main()