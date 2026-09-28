"""
Measures how long the ORIGINAL scripted ghost AI (Baseline 1, untouched)
takes to catch a player, starting from the real spawn positions. This
gives us a real-world floor/ceiling for choosing MAX_STEPS_PER_EPISODE in
the RL environment -- no RL involved here at all.

Run from the repo root:
    python measure_catch_time.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

from pacman import Game
from model.direction import Direction

NUM_EPISODES = 10
MAX_TICKS_PER_EPISODE = 5000  # generous ceiling just for this measurement
DIRECTIONS = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]


def run_one_episode():
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()  # skip the READY! wait

    starting_lives = engine.player.lives

    for tick in range(1, MAX_TICKS_PER_EPISODE + 1):
        engine.direction_command = random.choice(DIRECTIONS)
        engine.tick()  # ghosts use their normal follow_target(), no override

        if engine.player.lives < starting_lives:
            return tick  # caught!

    return None  # never caught within the ceiling


def main():
    catch_times = []

    for ep in range(1, NUM_EPISODES + 1):
        result = run_one_episode()
        if result is not None:
            print(f"Episode {ep}: caught at tick {result}")
            catch_times.append(result)
        else:
            print(f"Episode {ep}: NOT caught within {MAX_TICKS_PER_EPISODE} ticks")

    print("\n--- Results ---")
    if catch_times:
        print(f"Episodes with a catch: {len(catch_times)}/{NUM_EPISODES}")
        print(f"Min ticks to catch:    {min(catch_times)}")
        print(f"Max ticks to catch:    {max(catch_times)}")
        print(f"Average ticks to catch:{sum(catch_times)/len(catch_times):.1f}")
    else:
        print("No catches occurred in any episode -- investigate further before trusting episode length numbers.")


if __name__ == "__main__":
    main()