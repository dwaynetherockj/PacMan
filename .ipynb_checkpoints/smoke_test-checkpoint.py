"""
Headless smoke test for the PacMan game.

Runs the real game loop (via the existing Game class) with no physical
display or audio device, using Pygame's built-in dummy drivers. This
proves the game can be stepped without a screen, which is required for
training on Google Colab.

Run from the repo root:
    python smoke_test.py
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

from pacman import Game
from model.direction import Direction

NUM_STEPS = 500
DIRECTIONS = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]


def main():
    print("Creating game (headless)...")
    game = Game()
    print("Game created successfully. No display was opened.")

    engine = game.game_engine
    steps_run = 0

    for step in range(NUM_STEPS):
        engine.direction_command = random.choice(DIRECTIONS)
        engine.tick()
        steps_run = step + 1

        if engine.game_over:
            print(f"Game over at step {steps_run} (random play died fast, that's expected).")
            break

    print("\n--- Smoke test results ---")
    print(f"Steps run:      {steps_run}")
    print(f"Final score:    {engine.level.score}")
    print(f"Player lives:   {engine.player.lives}")
    print(f"Game over flag: {engine.game_over}")
    print("Smoke test PASSED: game stepped headlessly with no display window.")


if __name__ == "__main__":
    main()