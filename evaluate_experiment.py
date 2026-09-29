"""
Evaluate trained ghosts against random and scripted Blinky, all solo
(the other three ghosts do not move), on identical episodes.

Baselines only:
    python -u evaluate_experiment.py --tag baselines
With models ("name:nav" means trained with the direction flags):
    python -u evaluate_experiment.py --tag base_vs_nav \
        --models solo_base solo_base_s1 solo_base_s2 \
                 solo_nav:nav solo_nav_s1:nav solo_nav_s2:nav

Every condition uses the same seeds, so spawn tiles and player moves match.
Results are printed and saved to results_<tag>.csv
"""

import argparse
import csv
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

import pygame
pygame.time.wait = lambda ms: 0  # skip the game's 500 ms freeze on each catch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv, MAX_STEPS_PER_EPISODE
from rl_env.maze_distance import REAL_BLINKY_DISTANCE

DISTANCES = [3, 6, 9, 13, 16, REAL_BLINKY_DISTANCE]


def load_model(spec):
    name, _, flag = spec.partition(":")
    nav = flag == "nav"
    model = PPO.load(name)
    dummy = DummyVecEnv([lambda: PacmanGhostEnv(solo_ghost=True, nav_obs=nav)])
    norm = VecNormalize.load(f"{name}_stats.pkl", dummy)
    norm.training = False
    norm.norm_reward = False
    return name, nav, model, norm


def run_episode(env, policy, seed):
    """Returns steps to catch, or None if the ghost never caught the player."""
    random.seed(seed)                     # same spawn tile and player moves in every condition
    rng = random.Random(seed + 10_000)    # separate stream for the random baseline's actions
    obs, _ = env.reset()
    while True:
        kind = policy[0]
        if kind == "model":
            _, model, norm = policy
            action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
            action = int(action)
        elif kind == "random":
            action = rng.randrange(4)
        else:
            action = 0  # ignored: the scripted ghost decides for itself
        obs, _, terminated, truncated, info = env.step(action)
        if terminated:
            return env.step_count
        if truncated:
            return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--tag", default="eval")
    a = p.parse_args()

    # (label, uses direction flags, policy)
    conditions = [("random", False, ("random",)), ("scripted", False, ("scripted",))]
    for spec in a.models:
        name, nav, model, norm = load_model(spec)
        conditions.append((name, nav, ("model", model, norm)))

    rows = []
    for d in DISTANCES:
        print(f"\n=== Spawn distance {d} tiles ({a.episodes} episodes each) ===")
        for label, nav, policy in conditions:
            env = PacmanGhostEnv(curriculum_distance=d, solo_ghost=True,
                                 scripted_trained_ghost=(policy[0] == "scripted"),
                                 nav_obs=nav)
            steps = [run_episode(env, policy, seed=ep) for ep in range(a.episodes)]
            hits = [s for s in steps if s is not None]
            rate = 100 * len(hits) / a.episodes
            mean_hit = sum(hits) / len(hits) if hits else float("nan")
            mean_all = sum(s if s is not None else MAX_STEPS_PER_EPISODE
                           for s in steps) / a.episodes
            print(f"{label:<14} catch rate {rate:5.1f}%   steps to catch {mean_hit:6.1f} "
                  f"(hits only)   {mean_all:6.1f} (misses count as {MAX_STEPS_PER_EPISODE})")
            rows.append([label, d, a.episodes, len(hits), round(rate, 1),
                         round(mean_hit, 1), round(mean_all, 1)])

    with open(f"results_{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "distance", "episodes", "catches", "catch_rate_pct",
                    "mean_steps_hits_only", "mean_steps_misses_as_max"])
        w.writerows(rows)
    print(f"\nSaved results_{a.tag}.csv")


if __name__ == "__main__":
    main()