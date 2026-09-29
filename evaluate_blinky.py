"""
Compare the trained Blinky against scripted Blinky under identical conditions.

Three conditions per spawn distance:
  1. RL Blinky alone           (other 3 ghosts do not move)
  2. Scripted Blinky alone     (other 3 ghosts do not move)
  3. RL Blinky + 3 scripted    (the training condition; shows who made each catch)

Every episode uses the same random seed across conditions, so the random
player moves and spawn tile match (a paired comparison).

Run from the repo root:
    python -u evaluate_blinky.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
from collections import Counter

import pygame
pygame.time.wait = lambda ms: 0  # skip the game's 500 ms freeze on each catch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv
from rl_env.maze_distance import REAL_BLINKY_DISTANCE

MODEL_PATH = "ppo_blinky_curriculum"
STATS_PATH = "vecnormalize_curriculum_stats.pkl"
EPISODES_PER_CELL = 50
DISTANCES = [3, 6, 9, 13, 16, REAL_BLINKY_DISTANCE]

# (name, env options, uses the trained model?)
CONDITIONS = [
    ("RL Blinky alone",        dict(solo_ghost=True,  scripted_trained_ghost=False), True),
    ("Scripted Blinky alone",  dict(solo_ghost=True,  scripted_trained_ghost=True),  False),
    ("RL Blinky + 3 scripted", dict(solo_ghost=False, scripted_trained_ghost=False), True),
]


def load_normalizer():
    dummy = DummyVecEnv([lambda: PacmanGhostEnv()])
    norm = VecNormalize.load(STATS_PATH, dummy)
    norm.training = False
    norm.norm_reward = False
    return norm


def run_episode(env, model, norm, seed):
    """Returns (steps_to_catch, catcher_name), or (None, None) if no catch."""
    random.seed(seed)
    obs, _ = env.reset()
    while True:
        if model is not None:
            action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
            action = int(action)
        else:
            action = 0  # ignored: the scripted ghost decides for itself
        obs, _, terminated, truncated, info = env.step(action)
        if terminated:
            return env.step_count, info.get("caught_by")
        if truncated:
            return None, None


def main():
    model = PPO.load(MODEL_PATH)
    norm = load_normalizer()

    probe = PacmanGhostEnv()
    probe.reset()
    print(f"The trained ghost (index 0) is: {type(probe.ghost).__name__}")

    for d in DISTANCES:
        print(f"\n=== Spawn distance {d} tiles ({EPISODES_PER_CELL} episodes each) ===")
        for name, kwargs, use_model in CONDITIONS:
            env = PacmanGhostEnv(curriculum_distance=d, **kwargs)
            steps_list = []
            catchers = Counter()
            for ep in range(EPISODES_PER_CELL):
                steps, catcher = run_episode(env, model if use_model else None, norm, seed=ep)
                if steps is not None:
                    steps_list.append(steps)
                    catchers[catcher] += 1
            n = len(steps_list)
            rate = 100 * n / EPISODES_PER_CELL
            avg = sum(steps_list) / n if n else float("nan")
            print(f"{name:<24} catch rate {rate:5.1f}%   avg steps to catch {avg:6.1f}   "
                  f"caught by {dict(catchers)}")


if __name__ == "__main__":
    main()