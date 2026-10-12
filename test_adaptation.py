"""
The key demonstration: a player who starts WEAK then becomes STRONG partway
through, to show the controller follows the change and raises difficulty.

The player plays weak for the first 5 lives, then strong for the next 5.
We give extra lives for this test only, so we see the full arc. Each life
lost, the controller guesses the player's skill and sets ghost speed.

Run from the repo root:
    python -u test_adaptation.py
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

WEAK_SKILL = 0.0
STRONG_SKILL = 1.0
LIVES_PER_PHASE = 5
SEED = 0
MAX_TICKS = 40000


def main():
    random.seed(SEED)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()

    # Give extra lives for this test so we see the full weak->strong arc.
    engine.player.lives = LIVES_PER_PHASE * 2 - 1   # game ends at lives == -1

    bot = SkillBot(WEAK_SKILL, seed=SEED)
    controller = AdaptiveController(engine)

    print(f"Player plays WEAK for {LIVES_PER_PHASE} lives, "
          f"then STRONG for {LIVES_PER_PHASE} lives.\n")

    lives_seen = engine.player.lives
    lives_lost = 0
    switched = False
    guesses = []   # (phase, guess)

    for t in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()
        controller.update()

        if engine.player.lives < lives_seen:
            lives_seen = engine.player.lives
            lives_lost += 1
            phase = "weak" if not switched else "strong"
            band = controller.current_band
            mult = BAND_SETTINGS[band]["ghost_speed_mult"]
            guesses.append((phase, band))
            print(f"Life {lives_lost:2d} ({phase:<6}) lost -> "
                  f"controller now guessing {band:<9} (ghost speed {mult}x)")

            # After the weak phase, switch the player to strong.
            if lives_lost == LIVES_PER_PHASE and not switched:
                bot = SkillBot(STRONG_SKILL, seed=SEED + 1)
                switched = True
                print("   --- PLAYER SWITCHED TO STRONG ---")

        if engine.game_over:
            break

    print("\nSummary:")
    weak_guesses = [g for ph, g in guesses if ph == "weak"]
    strong_guesses = [g for ph, g in guesses if ph == "strong"]
    print(f"  While playing WEAK,   controller guessed: {weak_guesses}")
    print(f"  While playing STRONG, controller guessed: {strong_guesses}")


if __name__ == "__main__":
    main()